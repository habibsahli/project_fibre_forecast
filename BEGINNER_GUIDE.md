# 🎓 Beginner's Guide: How This ETL Pipeline Works

## 📖 Table of Contents
1. [What Does This Project Do?](#what-does-this-project-do)
2. [The Big Picture](#the-big-picture)
3. [Step-by-Step Explanation](#step-by-step-explanation)
4. [How Data Quality is Ensured](#how-data-quality-is-ensured)
5. [The Database Structure Explained](#the-database-structure-explained)
6. [Real-World Example](#real-world-example)

---

## What Does This Project Do?

### 🎯 **Simple Answer**

Imagine you have a messy Excel file with thousands of fiber optic subscription records. This project:

1. **Reads** the messy CSV file
2. **Cleans** all the data (fixes phone numbers, dates, removes duplicates)
3. **Organizes** it into a smart database structure
4. **Makes it easy** to answer questions like:
   - "How many subscriptions happened in Tunis last month?"
   - "Which dealer sold the most packages?"
   - "What's our most popular fiber plan?"

### 🏭 **Real-World Analogy**

Think of it like a **factory production line**:

```
Raw Materials → Inspection → Cleaning → Assembly → Finished Product
    (CSV)     → (Validate) → (Fix)    → (Organize) → (Database)
```

---

## The Big Picture

### 📊 **The Problem We're Solving**

**Before (Messy Data):**
```
❌ Phone numbers in different formats: "0098765432", "21698765432", "98765432"
❌ Dates in multiple formats: "01/15/2024", "15-01-2024", "2024/01/15"
❌ Duplicate records
❌ Missing information
❌ Hard to analyze or get insights
```

**After (Clean Database):**
```
✅ All phone numbers standardized: "21698765432"
✅ All dates in consistent format
✅ No duplicates
✅ Organized in a smart way
✅ Easy to query: "SELECT COUNT(*) FROM subscriptions WHERE city='Tunis'"
```

### 🔄 **The Three Phases (ETL)**

```
E.T.L. = Extract → Transform → Load

Think of it as(in chunks):
E = COLLECT the ingredients
T = PREPARE and CLEAN them
L = SERVE the finished dish
```

---

## Step-by-Step Explanation

### 🔍 **Phase 1: EXTRACT (Collecting the Data)**

**What happens:**
1. The program looks in a folder called `data/landing/`
2. It finds your CSV file (like "subscriptions_2024.csv")
3. It reads the file into computer memory
4. It checks: "Does this file have all the required columns?"

**Required columns:**
- MSISDN (phone number)
- KIT_CODE (equipment ID)
- DEALER_ID (who sold it)
- OFFRE (package name)
- CITY, GOVERNORATE (location)
- CREATION_DATE (when the subscription was made)
- And a few more...

**Real-world example:**
```
You → Drop file in landing/ folder
Program → "Found 1 file: subscriptions_2024.csv"
Program → "Reading file... 3,000 rows found"
Program → "Checking columns... ✓ All 13 required columns present"
Program → "Saving a backup copy to raw/ folder for safety"
```

**Why backup the raw file?**
- Like keeping the original receipt
- If something goes wrong, we can always go back
- For audit purposes (proving we didn't lose data)

---

### 🔧 **Phase 2: TRANSFORM (Cleaning the Data)**

This is where the magic happens! The program cleans and validates every single record.

#### **Step 2.1: Fix Phone Numbers (MSISDN)**

**The Problem:**
People enter phone numbers in different ways:
```
❌ "0098765432"      (has extra 00)
❌ "98765432"        (missing country code)
❌ "21698765432"     (correct!)
❌ "216-98-765-432"  (has dashes)
```

**What the program does:**
```python
1. Remove all non-numbers (dashes, spaces, etc.)
2. Check the format:
   - Starts with "216"? → Keep it
   - Starts with "0"? → Replace with "216"
   - 8 digits only? → Add "216" at the start
3. Final check: Must be exactly 12 digits starting with "216"
```

**Result:** All phone numbers look like `21698765432` ✅

---

#### **Step 2.2: Parse and Validate Dates**

**The Problem:**
Dates can be written many ways:
```
MM/DD/YYYY HH:MM:SS  →  "01/15/2024 10:30:00"
DD/MM/YYYY HH:MM:SS  →  "15/01/2024 10:30:00"
YYYY-MM-DD HH:MM:SS  →  "2024-01-15 10:30:00"
```

**What the program does:**
```python
1. Try format #1 (MM/DD/YYYY) → If it works, great!
2. If not, try format #2 (DD/MM/YYYY)
3. If not, try format #3 (YYYY-MM-DD)
4. If none work → Mark this record as invalid
5. Check year is between 2020-2026 (reasonable range)
```

**Result:** All dates are properly parsed and validated ✅

---

#### **Step 2.3: Clean Text Fields**

**What the program does:**
```python
# Cities and locations → Title Case
"el menzah" → "El Menzah"
"LA SOUKRA" → "La Soukra"

# IDs and codes → UPPERCASE
"m18" → "M18"
"kit001" → "KIT001"

# Remove extra spaces
"  Tunis  " → "Tunis"
```

**Why?** So "Tunis", "tunis", and "TUNIS" are all treated as the same city!

---

#### **Step 2.4: Validate GPS Coordinates**

**The Logic:**
Tunisia is located at:
- Latitude: between 30° and 38° North
- Longitude: between 7° and 12° East

**What the program does:**
```python
If coordinates are present:
    Check if lat is between 30 and 38
    Check if lon is between 7 and 12
    If outside these bounds → Set to NULL (not reject the whole record)
```

**Why not reject?** GPS is optional. Better to keep the subscription data even if GPS is wrong.

---

#### **Step 2.5: Remove Duplicates**

**What the program does:**
```python
1. Look at all records
2. Group by MSISDN (phone number is unique)
3. If same phone number appears twice → Keep only the FIRST one
4. Count how many duplicates were removed
```

**Example:**
```
Record 1: MSISDN=21698765432, Date=2024-01-15  ← KEEP
Record 2: MSISDN=21698765432, Date=2024-01-16  ← REMOVE (duplicate)
Record 3: MSISDN=21698765433, Date=2024-01-15  ← KEEP (unique)
```

---

#### **Step 2.6: Validation Summary**

For each record, the program decides:

**✅ VALID** if it has:
- Valid phone number (MSISDN)
- Valid date (CREATION_DATE)
- Valid equipment code (KIT_CODE)

**❌ INVALID** if it's missing any critical field

**Example:**
```
Record A: 
  MSISDN = "21698765432" ✓
  DATE = "2024-01-15" ✓
  KIT_CODE = "KIT001" ✓
  → VALID ✅

Record B:
  MSISDN = "invalid" ✗
  DATE = "2024-01-15" ✓
  KIT_CODE = "KIT001" ✓
  → INVALID ❌ (bad phone number)
```

---

### 📥 **Phase 3: LOAD (Organizing into Database)**

Now we have clean data. Time to organize it smartly!

#### **Understanding the Star Schema**

**Simple Analogy:**

Imagine you're organizing a library:
- Instead of writing author name, book title, date on EVERY card
- You create:
  - An AUTHORS table (author names only)
  - A DATES table (all dates)
  - A MAIN TABLE that just points to these

**Why?** Saves space and makes queries faster!

---

#### **Step 3.1: Load Raw Data (Archive)**

```python
1. Save ALL records (even invalid ones) to "raw_data" table
2. This is for auditing: "We received exactly 3,000 records"
```

**Why?** Proof that we didn't lose or hide any data.

---

#### **Step 3.2: Load Clean Data**

```python
1. Save cleaned records to "clean_data" table
2. Valid ones marked as is_valid=TRUE
3. Invalid ones marked as is_valid=FALSE with error message
```

**Example:**
```
Record | MSISDN        | is_valid | validation_errors
-------|---------------|----------|-------------------
1      | 21698765432   | TRUE     | NULL
2      | invalid       | FALSE    | "MSISDN: Invalid format"
```

---

#### **Step 3.3: Create Dimension Tables**

These are like "reference books" in our library.

**1. dim_temps (Time Dimension)**

Instead of storing "2024-01-15" everywhere, we:
```python
1. Generate ALL dates from 2024-2026 (about 1,096 dates)
2. For each date, calculate:
   - Day of week (Monday = 0, Sunday = 6)
   - Month name ("January")
   - Quarter (Q1, Q2, Q3, Q4)
   - Is it a weekend? (TRUE/FALSE)
   - Is it a holiday? (TRUE/FALSE)

Then in our main table, we just store: date_id = 523 (instead of full date)
```

**Why?** 
- Saves space (1 number vs. 20 characters)
- Makes queries easier: "Find all weekend subscriptions"
- Pre-calculated info: quarter, holiday status, etc.

---

**2. dim_offres (Offers Dimension)**

Store all unique package names:
```
offre_id | nom_offre              | categorie
---------|------------------------|----------
1        | Pack Fibre Villa 50M   | Villa
2        | Fibre Pro              | Pro
3        | Pack Dual Play 20M     | Standard
```

Instead of repeating "Pack Fibre Villa 50M" 500 times, we just store `offre_id = 1`

---

**3. dim_geographie (Geography Dimension)**

Store all unique locations:
```
geo_id | city      | governorate | latitude | longitude
-------|-----------|-------------|----------|----------
1      | El Menzah | Tunis       | 36.8500  | 10.1900
2      | La Soukra | Ariana      | 36.8520  | 10.1950
```

---

**4. dim_dealers (Dealers Dimension)**

Store all unique vendors:
```
dealer_id_pk | dealer_id | dealer_name
-------------|-----------|-------------
1            | M18       | Dealer_M18
2            | S40       | Dealer_S40
```

---

#### **Step 3.4: Create Fact Table (The Main Table)**

This is the **heart** of our database. It connects everything:

```
fact_abonnements (Subscriptions Fact Table)
----------------------------------------------
abonnement_id | msisdn        | kit_code | date_id | offre_id | geo_id | dealer_id_pk
--------------|---------------|----------|---------|----------|--------|-------------
1             | 21698765432   | KIT001   | 523     | 1        | 15     | 4
2             | 21698765433   | KIT002   | 524     | 2        | 16     | 5
```

**Notice:**
- Instead of storing full date → Just `date_id = 523`
- Instead of "Pack Fibre Villa 50M" → Just `offre_id = 1`
- Instead of "El Menzah, Tunis, 36.85, 10.19" → Just `geo_id = 15`

**This is the STAR SCHEMA!**

```
         dim_temps (when?)
              │
              │
    ┌─────────┼─────────┬─────────┐
    │         │         │         │
dim_offres  │    dim_geographie  dim_dealers
(what?)     │    (where?)        (who?)
            │
            │
      fact_abonnements
      (the main story)
```

---

#### **Step 3.5: Validate Everything Works**

Final quality checks:
```python
1. Count records in fact_abonnements
2. Check: Do all date_ids exist in dim_temps? ✓
3. Check: Do all offre_ids exist in dim_offres? ✓
4. Check: Do all geo_ids exist in dim_geographie? ✓
5. Check: Are there any duplicate phone numbers? ✗ (should be 0)
```

If all checks pass → ✅ **SUCCESS!**

---

## How Data Quality is Ensured

### 🛡️ **Built-in Quality Checks**

#### **1. Validation at Every Step**

```
Raw Data → Validate columns exist ✓
         → Validate critical fields ✓
         → Validate format (phone, date) ✓
         → Validate ranges (GPS, year) ✓
         → Remove duplicates ✓
         → Final integrity check ✓
```

#### **2. Four Levels of Validation**

**Level 1: File Level**
- "Does the CSV have all 13 required columns?"
- If NO → Stop immediately, show error

**Level 2: Record Level (Critical Fields)**
- MSISDN: Must be valid Tunisian phone
- DATE: Must parse to valid date
- KIT_CODE: Must not be empty

**Level 3: Record Level (Optional Fields)**
- GPS: Validate if present, set NULL if invalid (don't reject)
- Locality: Can be empty

**Level 4: Database Level**
- Foreign keys must exist
- No orphaned records
- No null critical fields

---

#### **3. Error Tracking**

Every error is logged:

```python
Record 152: MSISDN: Invalid length: 10
Record 231: CREATION_DATE: Could not parse date: "32/15/2024"
Record 445: GPS: Latitude out of bounds: 45.2
```

At the end, you get a report:
```
Total records read: 3,000
Valid records: 2,945
Invalid records: 55
Duplicates removed: 12

Top errors:
- MSISDN invalid format: 23
- Date parse error: 18
- Missing KIT_CODE: 14
```

---

#### **4. Audit Trail**

Everything is logged for transparency:

**Raw Data Table:**
- Keeps EVERY record as it came in (proof)

**Clean Data Table:**
- Shows which records passed/failed validation
- Stores validation error messages

**Audit Log Table:**
- When was the ETL run?
- How many records processed?
- How long did it take?
- Any errors?

**Example Audit Log:**
```
Process: ETL Pipeline
Start: 2024-01-15 10:30:00
End: 2024-01-15 10:31:23
Duration: 83 seconds
Records read: 3,000
Records loaded: 2,945
Records rejected: 55
Status: SUCCESS ✓
```

---

## The Database Structure Explained

### 🌟 **Why Star Schema?**

**Problem with Flat Table:**
```
MSISDN        | DATE       | CITY      | GOVERNORATE | OFFRE              | DEALER
21698765432   | 2024-01-15 | El Menzah | Tunis       | Pack Fibre Villa   | M18
21698765433   | 2024-01-15 | El Menzah | Tunis       | Pack Fibre Villa   | M18
21698765434   | 2024-01-15 | El Menzah | Tunis       | Pack Fibre Villa   | M18
```

**Problems:**
1. "El Menzah, Tunis" repeated 1,000 times → Wastes space
2. "Pack Fibre Villa" repeated 500 times → Wastes space
3. Hard to update: "If we rename M18 to MaxSales, we need to update 500 rows!"
4. Slow queries: "Find all subscriptions in Tunis" needs to scan EVERY character

---

**Solution: Star Schema**

**Dimensions (Reference Tables):**
```sql
-- dim_geographie: Store each location ONCE
geo_id=15: El Menzah, Tunis

-- dim_offres: Store each offer ONCE
offre_id=1: Pack Fibre Villa

-- dim_dealers: Store each dealer ONCE
dealer_id_pk=4: M18

-- dim_temps: Store each date ONCE
date_id=523: 2024-01-15 (Monday, Q1, Not Weekend)
```

**Fact Table:**
```sql
MSISDN        | date_id | geo_id | offre_id | dealer_id_pk
21698765432   | 523     | 15     | 1        | 4           
21698765433   | 523     | 15     | 1        | 4
21698765434   | 523     | 15     | 1        | 4
```

**Benefits:**
1. ✅ Each location stored ONCE
2. ✅ Each offer stored ONCE  
3. ✅ Easy to update: Change dealer name in ONE place
4. ✅ Fast queries: Compare numbers (IDs) instead of text
5. ✅ Smaller database size
6. ✅ Can add more info to dimensions without touching facts

---

### 📊 **Querying Made Easy**

**Question:** "How many subscriptions in Tunis?"

**Flat Table Way (Slow):**
```sql
SELECT COUNT(*) 
FROM big_table 
WHERE governorate = 'Tunis'  -- Scans every character
```

**Star Schema Way (Fast):**
```sql
SELECT COUNT(*) 
FROM fact_abonnements f
JOIN dim_geographie g ON f.geo_id = g.geo_id
WHERE g.governorate = 'Tunis'  -- Just compares IDs (numbers)
```

**Why faster?** Comparing `geo_id = 15` is faster than comparing text "governorate = 'Tunis'"

---

## Real-World Example

Let's follow ONE subscription through the entire pipeline:

### 📋 **Raw Input (CSV)**

```csv
KIT_CODE,MSISDN,DEALER_ID,OFFRE,DEBIT,CITY,GOVERNORATE,POSTAL_CODE,LATITUDE,LONGITUDE,LOCALITY_NAME,DELEGATION_NAME,CREATION_DATE
KIT001,0098765432,m18,pack fibre villa,Pack Villa 50M,el menzah,TUNIS,1001,36.8500,10.1900,El Menzah,Tunis,01/15/2024 10:30:00
```

---

### 🔍 **Phase 1: EXTRACT**

```
✓ File found: subscriptions_2024.csv
✓ All required columns present
✓ Raw file archived to: data/raw/raw_data_20240115_103000.csv
✓ Record loaded into memory
```

---

### 🔧 **Phase 2: TRANSFORM**

**Step 1: Fix Phone Number**
```
Input:  "0098765432"
Remove non-digits: "0098765432"
Starts with 00: Remove → "98765432"
Only 8 digits: Add 216 → "21698765432"
Output: "21698765432" ✓
```

**Step 2: Parse Date**
```
Input: "01/15/2024 10:30:00"
Try format MM/DD/YYYY: ✓ Success!
Parsed: datetime(2024, 1, 15, 10, 30, 0)
Year check: 2024 is between 2020-2026 ✓
Output: 2024-01-15 10:30:00 ✓
```

**Step 3: Clean Text**
```
KIT_CODE: "KIT001" → "KIT001" (already uppercase) ✓
DEALER_ID: "m18" → "M18" (uppercase) ✓
CITY: "el menzah" → "El Menzah" (title case) ✓
GOVERNORATE: "TUNIS" → "Tunis" (title case) ✓
OFFRE: "pack fibre villa" → "Pack Fibre Villa" ✓
```

**Step 4: Validate GPS**
```
Latitude: 36.8500
Check: Is 30 ≤ 36.85 ≤ 38? ✓ YES
Longitude: 10.1900
Check: Is 7 ≤ 10.19 ≤ 12? ✓ YES
Output: Valid GPS ✓
```

**Step 5: Overall Validation**
```
✓ MSISDN: Valid
✓ DATE: Valid
✓ KIT_CODE: Valid
→ Record is VALID ✅
```

---

### 📥 **Phase 3: LOAD**

**Step 1: Save to raw_data (archive)**
```sql
INSERT INTO raw_data VALUES (
  'KIT001', '0098765432', 'm18', ...  -- Original as received
)
```

**Step 2: Save to clean_data**
```sql
INSERT INTO clean_data VALUES (
  'KIT001', '21698765432', 'M18', ...  -- Cleaned version
)
```

**Step 3: Build Dimensions**

**dim_temps:**
```sql
-- Check if date 2024-01-15 exists
-- If not, insert:
INSERT INTO dim_temps VALUES (
  date_id: 523,
  full_date: 2024-01-15,
  day_of_week: 0 (Monday),
  month: 1,
  month_name: 'January',
  quarter: 1,
  year: 2024,
  is_weekend: FALSE,
  is_holiday: FALSE
)
```

**dim_offres:**
```sql
-- Check if "Pack Fibre Villa" exists
-- If not, insert:
INSERT INTO dim_offres VALUES (
  offre_id: 1,
  nom_offre: 'Pack Fibre Villa',
  categorie: 'Villa'
)
```

**dim_geographie:**
```sql
-- Check if "El Menzah, Tunis" exists
-- If not, insert:
INSERT INTO dim_geographie VALUES (
  geo_id: 15,
  city: 'El Menzah',
  governorate: 'Tunis',
  postal_code: 1001,
  latitude: 36.8500,
  longitude: 10.1900
)
```

**dim_dealers:**
```sql
-- Check if "M18" exists
-- If not, insert:
INSERT INTO dim_dealers VALUES (
  dealer_id_pk: 4,
  dealer_id: 'M18'
)
```

**Step 4: Load Fact**

```sql
-- Look up IDs:
date_id = 523 (from dim_temps where date = 2024-01-15)
offre_id = 1 (from dim_offres where name = 'Pack Fibre Villa')
geo_id = 15 (from dim_geographie where city = 'El Menzah')
dealer_id_pk = 4 (from dim_dealers where dealer_id = 'M18')

-- Insert fact record:
INSERT INTO fact_abonnements VALUES (
  msisdn: '21698765432',
  kit_code: 'KIT001',
  date_id: 523,
  offre_id: 1,
  geo_id: 15,
  dealer_id_pk: 4,
  debit: 'Pack Villa 50M'
)
```

---

### ✅ **DONE! Now You Can Query**

```sql
-- Find this subscription:
SELECT 
  f.msisdn,
  t.full_date,
  g.city,
  g.governorate,
  o.nom_offre,
  d.dealer_id
FROM fact_abonnements f
JOIN dim_temps t ON f.date_id = t.date_id
JOIN dim_geographie g ON f.geo_id = g.geo_id
JOIN dim_offres o ON f.offre_id = o.offre_id
JOIN dim_dealers d ON f.dealer_id_pk = d.dealer_id_pk
WHERE f.msisdn = '21698765432'
```

**Result:**
```
msisdn        | full_date  | city      | governorate | nom_offre        | dealer_id
21698765432   | 2024-01-15 | El Menzah | Tunis       | Pack Fibre Villa | M18
```

---

## 🎯 Summary: Why This Approach Works

### ✅ **For Data Quality:**
1. **Validation at every step** catches errors early
2. **Standardization** ensures consistency
3. **Audit trail** provides transparency
4. **Error tracking** helps improve processes

### ✅ **For Performance:**
1. **Star schema** makes queries fast
2. **Indexed IDs** speed up joins
3. **Pre-calculated dimensions** (like day of week) save time
4. **Normalized data** reduces storage

### ✅ **For Maintainability:**
1. **Modular code** - each phase separate
2. **Clear separation** - Extract, Transform, Load
3. **Comprehensive logging** aids debugging
4. **Good documentation** helps new developers

### ✅ **For Business:**
1. **Easy to query** - answer questions quickly
2. **Historical data** - raw data preserved
3. **Scalable** - can handle millions of records
4. **Automated** - runs daily without intervention

---

## 🚀 Next Steps for Learning

1. **Run the pipeline** with sample data (see QUICKSTART.md)
2. **Watch the logs** to see each step happen
3. **Query the database** to explore the results
4. **Try modifying** validation rules in config.py
5. **Add your own** validation logic

---

**Remember:** The best way to learn is by doing. Don't be afraid to experiment!

---

Need help? Check:
- **QUICKSTART.md** - Get started in 5 minutes
- **README.md** - Complete reference
- **INSTALLATION_GUIDE.md** - Detailed setup

Happy learning! 🎓
