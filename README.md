# Enhanced Earthquake Phase Picking by Denoising Distributed Acoustic Sensing (DAS)

This project is a novel practice that combines denoising and phase picking on DAS and 
integrate seismic networks with DAS to detect earthquakes in the offshore environment. 

The metadata for a Python project is defined in the `pyproject.toml` file,
an example of which is included in this project.

----

The basic workflow is in three steps.

1. Pre-process the DAS data to form 2-dimensional space-time images. [notebook00]
2. Denoise the DAS images using a pre-trained denoiser, which are saved in the ["models"] folder. [notebook01]
3. Use the ELEP tool to pick P and S phases in batch. [notebook01]
4. Use PyOcto to associate the picks of DAS and seismic networks.[notebook03]



[DAS denoising]: https://github.com/Denolle-Lab/DASdenoise
[ELEP]: https://github.com/congcy/ELEP
[PyOcto]: https://github.com/yetinam/pyocto
[notebook00]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/paper0_dataprep_akdas.ipynb
[notebook01]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/paper1_denoise_pick.ipynb
[notebook02]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/paper2_phase_pick_stats.ipynb
[notebook03]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/blob/main/examples/paper3_catalog_building.ipynb
["models"]: https://github.com/Denolle-Lab/Shi_etal_2023_denoiseDAS/tree/main/models/mlmodel
