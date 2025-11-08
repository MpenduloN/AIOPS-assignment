def sort_list_of_dicts(list_of_dicts, key_to_sort_by):
  """
  Sorts a list of dictionaries by a specified key.

  Args:
    list_of_dicts: A list of dictionaries.
    key_to_sort_by: The key in the dictionaries to sort by.

  Returns:
    A new list of dictionaries, sorted by the specified key.
  """
  return sorted(list_of_dicts, key=lambda x: x[key_to_sort_by])

# Example usage:
data = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]

sorted_data = sort_list_of_dicts(data, 'age')
print(sorted_data)