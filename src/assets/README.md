# Hurtlex data (modified)

This folder contains **modified and trimmed** versions of the English and Italian Hurtlex lexicons.

- Original: https://github.com/valeriobasile/hurtlex
  (files `lexica/EN/1.2/hurtlex_EN.tsv` and `lexica/IT/1.2/hurtlex_IT.tsv`)
- Authors: Elisa Bassignana, Valerio Basile, Viviana Patti
- License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Non-commercial use only. Derived files must keep the same license.

## Modifications

The same changes were applied to both lexicons:

- **Trimmed to a curated subset of entries.** No entries were added: every `id` in
  the modified files exists in the original.
- **Columns removed:** `pos`, `category` and `stereotype`. The modified files only
  keep `id`, `lemma` and `level`.
- **`level` value changed for some entries** (listed below).
- All other `lemma` and `level` values are unchanged from the original.

| Language | Original entries | Entries kept | `level` changed |
|---|---|---|---|
| English (EN) | 8,228 | 115 (about 1.4%) | 7 |
| Italian (IT) | 6,940 | 75 (about 1.1%) | 17 |

### English: `level` changes

| id | lemma | original | modified |
|---|---|---|---|
| EN1918 | retard | inclusive | conservative |
| EN1320 | whoreson | inclusive | conservative |
| EN4450 | whoremongers | inclusive | conservative |
| EN3772 | whoremonger | inclusive | conservative |
| EN4431 | whoremongering | inclusive | conservative |
| EN4454 | whore-monger | inclusive | conservative |
| EN6671 | gyp | conservative | inclusive |

### Italian: `level` changes

| id | lemma | original | modified |
|---|---|---|---|
| IT669 | zoccola | inclusive | conservative |
| IT1652 | finocchio | inclusive | conservative |
| IT4177 | cretino | inclusive | conservative |
| IT357 | imbecille | inclusive | conservative |
| IT1540 | cazzo | inclusive | conservative |
| IT1647 | frocio | inclusive | conservative |
| IT1896 | fica | inclusive | conservative |
| IT3630 | negro | inclusive | conservative |
| IT1599 | cazzeggiare | inclusive | conservative |
| IT352 | minchione | conservative | inclusive |
| IT342 | deficiente | conservative | inclusive |
| IT2194 | zingaro | conservative | inclusive |
| IT3868 | minchionare | conservative | inclusive |
| IT2340 | handicappato | conservative | inclusive |
| IT1477 | pederasta | conservative | inclusive |
| IT676 | puttana | conservative | inclusive |
| IT1660 | invertito | conservative | inclusive |

## Citation

Bassignana, E., Basile, V., Patti, V. (2018). Hurtlex: A Multi-lingual
Lexicon of Words to Hurt. CLiC-it 2018.

## License
Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode