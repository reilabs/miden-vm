
## std::collections::mmr
| Procedure | Description |
| ----------- | ------------- |
| u32unchecked_trailing_ones | Computes trailing number of ones in `number`.<br /><br />Stack transition:<br /><br />Input: [number, ...]<br /><br />Output: [trailing_ones, ...]<br /><br />Cycles: 6 + 11 * trailing_ones |
| trailing_ones | Computes trailing number of ones in `number`.<br /><br />Stack transition:<br /><br />Input: [number, ...]<br /><br />Output: [trailing_ones, ...]<br /><br />Cycles:<br /><br />- 13 + 11 * trailing_ones, when fewer than 32 trailing true bits<br /><br />- 18 + 11 * trailing_ones, when more than 32 traling true bits |
| ilog2_checked | Computes the `ilog2(number)` and `2^(ilog2(number))`.<br /><br />number must be non-zero, otherwise this will error<br /><br />Stack transition:<br /><br />Input: [number, ...]<br /><br />Output: [ilog2, power_of_two, ...]<br /><br />Cycles:  12 + 9 * leading_zeros |
| get | Loads the leaf at the absolute `pos` in the MMR.<br /><br />This MMR implementation supports only u32 positions.<br /><br />Stack transition:<br /><br />Input: [pos, mmr_ptr, ...]<br /><br />Output: [N, ...] where `N` is the leaf and `R` is the MMR peak that owns the leaf.<br /><br />Cycles: 65 + 9 * tree_position (where `tree_position` is 0-indexed bit position from most to least significant) |
| num_leaves_to_num_peaks | Given the num_leaves of a MMR returns the num_peaks.<br /><br />Input: [num_leaves, ...]<br /><br />Output: [num_peaks, ...]<br /><br />Cycles: 69 |
| num_peaks_to_message_size | Given the num_peaks of a MMR, returns the hasher state size after accounting<br /><br />for the required padding.<br /><br />Input: [num_peaks, ...]<br /><br />Output: [len, ...]<br /><br />Cycles: 17 |
| unpack | Load the MMR peak data based on its hash.<br /><br />Input: [HASH, mmr_ptr, ...]<br /><br />Output: [...]<br /><br />Where:<br /><br />- HASH: is the MMR peak hash, the hash is expected to be padded to an even<br /><br />length and to have a minimum size of 16 elements<br /><br />- The advice map must contain a key with HASH, and its value is<br /><br />`num_leaves \|\| hash_data`, and hash_data is the data used to computed `HASH`<br /><br />- mmt_ptr: the memory location where the MMR data will be written to,<br /><br />starting with the MMR forest (its total leaves count) followed by its peaks<br /><br />Cycles: 162 + 9 * extra_peak_pair cycles<br /><br />where `extra_peak` is the number of peak pairs in addition to the first<br /><br />16, i.e. `round_up((num_of_peaks - 16) / 2)` |
| pack | Computes the hash of the given MMR and copies it to the Advice Map using its hash as a key.<br /><br />Input: [mmr_ptr, ...]<br /><br />Output: [HASH, ...]<br /><br />Cycles: 128 + 3 * num_peaks |
| add | Adds a new element to the MMR.<br /><br />This will update the MMR peaks in the VM's memory and the advice provider<br /><br />with any merged nodes.<br /><br />Input: [EL, mmr_ptr, ...]<br /><br />Output: [...]<br /><br />Cycles: 108 + 46 * peak_merges |