from pokemon_filtering import identify_pokemon_by_criteria  
from pathlib import Path

base_dir = Path(__file__).parent.parent
pokeapi_data=base_dir/"Classification"/"pokeapi_data"


# How Many Mono Normal Pokemon?
normal_count, normal_df = identify_pokemon_by_criteria(
    generations=None, 
    typing=('Normal',), 
    order_exact=False, 
    loose=False, 
    data_dir=pokeapi_data)

# How Many Mono Water Pokemon?
water_count, water_df = identify_pokemon_by_criteria(
    generations=None, 
    typing=('Water',), 
    order_exact=False, 
    loose=False, 
    data_dir=pokeapi_data)

# Most Common Dual Type?
_, all_df = identify_pokemon_by_criteria(generations=None, typing=None, data_dir=pokeapi_data)
dual_df = all_df[all_df['type2'] != 'None'].copy()
dual_df['combo'] = dual_df.apply(lambda x: tuple(sorted([x['type1'], x['type2']])), axis=1)
dual_counts = dual_df['combo'].value_counts()

print(f"Most Common Dual Type: {dual_counts.index[0]} with {dual_counts.iloc[0]} Pokemon")

# Least Common Dual Type?
# There might be many tied for 1, let's find the ones with the minimum count
min_dual_count = dual_counts.min()
least_common_duals = dual_counts[dual_counts == min_dual_count].index.tolist()
print(f"Least Common Dual Type(s) (Count={min_dual_count}): {least_common_duals}")

# Least Common Mono Type?
mono_df = all_df[all_df['type2'] == 'None'].copy()
mono_counts = mono_df['type1'].value_counts()
min_mono_count = mono_counts.min()
least_common_monos = mono_counts[mono_counts == min_mono_count].index.tolist()
print(f"Least Common Mono Type(s) (Count={min_mono_count}): {least_common_monos}")
