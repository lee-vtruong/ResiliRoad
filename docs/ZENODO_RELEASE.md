# Creating the permanent software DOI

1. Sign in to Zenodo and connect the GitHub account that owns `lee-vtruong/ResiliRoad`.
2. In Zenodo's GitHub settings, enable the `ResiliRoad` repository.
3. On GitHub, create release `v1.0.0` from the final reviewed commit. Do not create
   the release until the manuscript numbers and repository outputs are frozen.
4. Wait for Zenodo to archive the release. Check title, author, ORCID, licence and
   files against `.zenodo.json` and `CITATION.cff`.
5. Copy the version DOI (not only the concept DOI) into the manuscript data/code
   statement and add a `[dataset]`/software entry to the reference list.
6. Recompile and visually inspect the PDF, then update `CITATION.cff` with the DOI.

Never invent or pre-fill a DOI. Zenodo assigns it only after a release is reserved
or deposited.
