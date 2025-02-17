def columns_calculation(lst):
    """
    Dynamically calculates the optimal number of columns for displaying symbols.

    It tries different column counts from max_columns down to 3 and selects the one
    that minimizes the remainder (ensuring even distribution).

    Parameters:
        lst (list): The list of symbols.

    Returns:
        int: The optimal number of columns.
    """
    max_columns = columns = 10  # Maximum number of columns allowed
    max_remainder = 0  # Tracks the highest remainder to optimize distribution

    # Try different column values from max_columns down to 3
    for columns_number in range(max_columns - 1, 3, -1):
        remainder = len(lst) % columns_number  # Compute remainder when dividing symbols

        # Choose the column count that maximizes remainder (ensures even layout)
        if remainder > max_remainder:
            columns = columns_number

    return columns  # Return the best column count


# Iterate through each category of symbols
for category, syms in symbols:
    # Sort symbols alphabetically (handling Unicode and variations correctly)
    syms = sorted(syms,
                  key=lambda sym: (render_symbol(sym, ignore_variant=True),
                                   sym.startswith(r"\var")),  # Prioritize variable-like symbols
                  reverse=(category == "Hebrew"))  # Hebrew is RTL, so reverse the order

    # Calculate the optimal number of columns dynamically
    columns = columns_calculation(syms)

    # Add category header to the table output
    lines.append('**%s**' % category)
    lines.append('')  # Empty line for spacing

    # Determine the maximum symbol width for formatting the table layout
    max_width = max(map(len, syms)) * 2 + 16  # Adjust width dynamically
