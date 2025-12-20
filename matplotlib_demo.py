import matplotlib.pyplot as plt

# --- Data ---
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [200, 300, 250, 400, 350]
expenses = [180, 250, 200, 300, 280]
products = ['Product A', 'Product B', 'Product C', 'Product D']
sales_distribution = [40, 25, 20, 15]

# --- Line Chart: Monthly Sales ---
plt.figure(figsize=(8, 5))
plt.plot(months, sales, marker='o', linestyle='-', color='b', label='Sales')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.savefig("line_chart.png")
plt.show()

# --- Bar Chart: Monthly Expenses ---
plt.figure(figsize=(8, 5))
plt.bar(months, expenses, color='orange', label='Expenses')
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Expenses")
plt.legend()
plt.savefig("bar_chart.png")
plt.show()

# --- Pie Chart: Product Sales Distribution ---
plt.figure(figsize=(6, 6))
plt.pie(sales_distribution, labels=products, autopct='%1.1f%%', startangle=140)
plt.title("Product Sales Distribution")
plt.savefig("pie_chart.png")
plt.show()
