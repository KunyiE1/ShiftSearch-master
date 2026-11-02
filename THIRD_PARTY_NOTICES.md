# Source and third-party notices

This release packages the ShiftSearch model, preprocessing, candidate
enumeration, search and training implementation used in the accompanying
manuscript.

The source workspace carries the GNU Affero General Public License, version 3.
The existing license text is preserved in `LICENSE`. Retain original notices
in reused source files and confirm the intended licensing of author additions
and weights before public release.

DeepSearch is a separate baseline described in
<https://www.nature.com/articles/s42256-024-00960-1>.
Its transformer implementation, pretrained weights and embedding databases are
not included in this repository.

PyTorch, Lightning, NumPy, pandas, LMDB, pyteomics, PyYAML and tqdm retain their
own licenses and are installed as dependencies. Protein Digestion Simulator
is an external tool; it is neither bundled nor reimplemented here.

Experimental spectra, proteomes and prepared pair tables have their own access
terms. Cite their data records separately and follow the providers' licenses.
