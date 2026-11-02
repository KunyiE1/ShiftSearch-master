# Before public release

- Confirm the public repository name and URL, then add the URL to the README
  and citation metadata. Local preparation does not create a remote repository.
- Confirm author/institution approval and source/weight licensing. Preserve
  the existing AGPL-3.0 license and inherited notices.
- Add the final manuscript or preprint bibliographic record when available.
  Do not invent a paper DOI, software DOI or data-release identifier.
- Provide persistent records for the prepared training pairs, spectrum-level
  splits, evaluation MGFs and search databases. The public spectral library
  alone does not supply all prepared experimental inputs.
- Verify that `weights/shiftsearch.ckpt` matches `weights/metadata.json`.
- Repeat the README installation in a clean Python 3.10 environment: select
  the CPU or CUDA 12.1 PyTorch build, install `requirements.txt` and this
  package, run `python -m pip check`, check all three CLI help commands,
  and run the synthetic CPU example and test suite.
- Inspect the validation record for the platforms and CPU/GPU configurations
  actually exercised. A configured CI workflow does not establish that its
  remote jobs have run.
- Upload only this repository. Exclude experimental datasets, search outputs,
  local environments, training logs, credentials and draft manuscripts.
- After choosing a release tag, archive the release in a persistent software
  repository if required by the manuscript's Code availability statement.
