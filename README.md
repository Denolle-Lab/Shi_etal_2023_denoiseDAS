# Enhanced Earthquake Phase Picking by Denoising Distributed Acoustic Sensing (DAS)

This project is a novel practice that combines [DAS denoising] and 
[Ensemble Learning for Earthquake Processing].

The metadata for a Python project is defined in the `pyproject.toml` file,
an example of which is included in this project.

----

The basic workflow is in three steps.

1. Pre-process the DAS data to form 2-dimensional space-time images. [notebook01]
2. Denoise the DAS images using a pre-trained denoiser, which are saved in the ["models"] folder. [notebook02]
3. Use the ELEP tool to pick P and S phases in batch. [notebook02]
4. Use PyOcto to associate the picks of DAS and seismic networks.[notebook03]



[DAS denoising]: https://github.com/Denolle-Lab/DASdenoise
[Ensemble Learning for Earthquake Processing]: https://github.com/congcy/ELEP
[notebook01]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/dataprep_akdas.ipynb
[notebook02]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/Denoise_and_pick.ipynb
["models"]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/tree/main/models
