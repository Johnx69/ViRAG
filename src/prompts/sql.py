SQL_GENERATION = """You are an expert SQL developer working with a Vietnamese sales database. 

{database_schema}

QUESTION TO ANSWER: {question}

INSTRUCTIONS:
1. Read the question carefully and identify what data is needed
2. Use ONLY the exact table and column names listed in the schema above
3. Follow the Vietnamese translation patterns provided
4. Use proper JOINs to get readable names (not just IDs)
5. Add appropriate GROUP BY, ORDER BY, and LIMIT clauses
6. Return ONLY the SQL query, no explanations

EXAMPLE TRANSLATIONS:
- "nhân viên nào bán được nhiều nhất" → Find employee with highest total revenue
- "doanh thu tháng X" → Sum revenue where month = X  
- "chi nhánh nào" → Which branch (need to join branches table)

Generate the SQL query now:
"""


SQL_FIX = """
URGENT SQL FIX NEEDED!

{database_schema}

FAILED QUERY: {sql_query}
ERROR MESSAGE: {error_message}

COMMON FIXES NEEDED:
1. Column name errors: Use exact names from schema (employee_name NOT first_name)
2. Table name errors: Use English names (employees NOT NhanVien)  
3. Missing JOINs: Add proper JOINs to get readable names
4. Date format errors: Use strftime for date filtering

ORIGINAL QUESTION: {question}

Generate a CORRECTED SQL query that:
1. Uses ONLY the exact table/column names from the schema
2. Includes proper JOINs for readable output
3. Handles the error mentioned above
4. Answers the original Vietnamese question

Return ONLY the corrected SQL query:
"""


SQL_SCHEMA = """
DATABASE: sell_data.sqlite
==========================

COMPLETE SCHEMA DOCUMENTATION:

### Table 1: customers
- customer_id (TEXT, PRIMARY KEY): Unique customer identifier (format: KH-XXXX)
- customer_name (TEXT): Full customer/company name in Vietnamese

EXAMPLE DATA:
KH-0001 | CÔNG TY TNHH ĐẦU TƯ PHÁT TRIỂN CÔNG NGHỆ BÁCH KHOA
KH-0002 | Công ty CP Phát triển Kỹ thuật và Thương mại Tân Đức
KH-0003 | NGƯỜI MUA KHÔNG LẤY HÓA ĐƠN

### Table 2: products  
- product_id (TEXT, PRIMARY KEY): Product code (format: P-XXX)
- product_name (TEXT): Product description in Vietnamese
- product_group (TEXT): Product category (mainly "Phần mềm" = Software)
- cost_price (INTEGER): Product cost in VND (no decimals)

EXAMPLE DATA:
P-001 | Phần mềm TeamViewer 12 Premium | Phần mềm | 17950000
P-003 | Phần mềm Win Home 10 64Bit Eng Intl 1pk DSP OEI DVD | Phần mềm | 1470000

### Table 3: employees
- employee_id (TEXT, PRIMARY KEY): Employee ID (format: NVXXX)  
- employee_name (TEXT): Full employee name in Vietnamese

EXAMPLE DATA:
NV001 | Nguyễn Nhật Tiến
NV003 | Đặng Minh Yến  
NV010 | Đặng Hữu Anh

### Table 4: branches
- branch_id (TEXT, PRIMARY KEY): Branch code (CN HN, CN HCM, CN DN)
- branch_name (TEXT): Full branch name in Vietnamese
- city (TEXT): City location

EXAMPLE DATA:
CN HN  | Chi nhánh Hà Nội       | Hà Nội
CN HCM | Chi nhánh Hồ Chí Minh  | Hồ Chí Minh  
CN DN  | Chi nhánh Đà Nẵng      | Đà Nẵng

### Table 5: kpi_targets
- year_month (TEXT): Format YYYY-MM (e.g., '2024-01', '2024-08')
- branch_id (TEXT): Foreign key to branches table
- kpi_value (INTEGER): Monthly KPI target in VND

EXAMPLE DATA:
2024-01 | CN HCM | 25000000000
2024-08 | CN HN  | 25000000000
2024-08 | CN HCM | 35000000000
2024-08 | CN DN  | 30000000000

### Table 6: sales (MAIN FACT TABLE)
- sale_id (INTEGER, PRIMARY KEY): Auto-increment ID
- accounting_date (TEXT): Date in YYYY-MM-DD format
- order_code (TEXT): Order reference (format: ĐHXXXXXXX)
- customer_id (TEXT): Foreign key to customers
- product_id (TEXT): Foreign key to products  
- quantity (INTEGER): Units sold
- unit_price (INTEGER): Price per unit in VND
- revenue (INTEGER): Total revenue = quantity × unit_price
- cost_of_goods_sold (INTEGER): Total cost of goods sold
- employee_id (TEXT): Foreign key to employees (who made the sale)
- branch_id (TEXT): Foreign key to branches (where sale was made)

EXAMPLE DATA:
1 | 2024-01-01 | ĐH3381365 | KH-0052 | P-396 | 10 | 1215000 | 12150000 | 8100000 | NV190 | CN HN
2 | 2024-08-15 | ĐH8038262 | KH-0257 | P-545 | 5  | 11655000 | 58275000 | 38850000 | NV164 | CN DN

==========================
CRITICAL SQL RULES:
==========================

1. **EXACT COLUMN NAMES** (Case-sensitive, no variations allowed):
   - customers: customer_id, customer_name
   - products: product_id, product_name, product_group, cost_price
   - employees: employee_id, employee_name  
   - branches: branch_id, branch_name, city
   - kpi_targets: year_month, branch_id, kpi_value
   - sales: sale_id, accounting_date, order_code, customer_id, product_id, quantity, unit_price, revenue, cost_of_goods_sold, employee_id, branch_id

2. **DATE HANDLING**:
   - accounting_date is TEXT in 'YYYY-MM-DD' format
   - For month filtering: strftime('%m', accounting_date) = 'XX'
   - For year filtering: strftime('%Y', accounting_date) = 'YYYY'
   - For year-month: strftime('%Y-%m', accounting_date) = 'YYYY-MM'

3. **VIETNAMESE QUERY TRANSLATIONS**:
   - "nhân viên" = employee → use employees table
   - "doanh thu" = revenue → use revenue column
   - "bán được nhiều nhất" = highest sales → SUM(revenue) DESC
   - "tháng X" = month X → strftime('%m', accounting_date) = 'X'
   - "chi nhánh" = branch → use branches table
   - "sản phẩm" = product → use products table
   - "khách hàng" = customer → use customers table

4. **COMMON JOINS NEEDED**:
   - Employee names: JOIN employees e ON s.employee_id = e.employee_id
   - Branch info: JOIN branches b ON s.branch_id = b.branch_id  
   - Product info: JOIN products p ON s.product_id = p.product_id
   - Customer info: JOIN customers c ON s.customer_id = c.customer_id

5. **AGGREGATION PATTERNS**:
   - Total revenue by employee: SELECT e.employee_name, SUM(s.revenue) FROM sales s JOIN employees e ON s.employee_id = e.employee_id GROUP BY e.employee_id, e.employee_name
   - Monthly revenue: SELECT strftime('%Y-%m', accounting_date) as month, SUM(revenue) FROM sales GROUP BY strftime('%Y-%m', accounting_date)
   - Top performers: ORDER BY SUM(revenue) DESC LIMIT 1

==========================
COMMON VIETNAMESE QUESTIONS & SQL PATTERNS:
==========================

Q: "Nhân viên nào bán được nhiều nhất?"
A: SELECT e.employee_name, SUM(s.revenue) as total_revenue 
   FROM sales s 
   JOIN employees e ON s.employee_id = e.employee_id 
   GROUP BY e.employee_id, e.employee_name 
   ORDER BY total_revenue DESC 
   LIMIT 1;

Q: "Doanh thu tháng 8 là bao nhiêu?"  
A: SELECT SUM(revenue) as total_revenue 
   FROM sales 
   WHERE strftime('%m', accounting_date) = '08';

Q: "Chi nhánh nào có doanh thu cao nhất?"
A: SELECT b.branch_name, SUM(s.revenue) as total_revenue
   FROM sales s
   JOIN branches b ON s.branch_id = b.branch_id
   GROUP BY b.branch_id, b.branch_name  
   ORDER BY total_revenue DESC
   LIMIT 1;

==========================
ERROR PREVENTION:
==========================

❌ NEVER use these (common mistakes):
- first_name, last_name (use employee_name)
- name (use customer_name, employee_name, product_name, branch_name)
- NhanVien, KhachHang (use English table names)
- price (use unit_price)
- total (use revenue)
- sales_amount (use revenue)

✅ ALWAYS use these exact names:
- Table names: customers, products, employees, branches, kpi_targets, sales
- Join keys: customer_id, product_id, employee_id, branch_id
- Revenue field: revenue (not total, not sales_amount)
- Date field: accounting_date (not date, not sale_date)
"""
