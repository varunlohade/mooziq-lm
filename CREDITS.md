# Credits

Every recorded sound in `site/public/samples/` comes from an openly licensed source. These files are not covered by the MIT licence of the code and models; each keeps the licence below. I checked each licence on its source page on 2026-11-20.

| Sound | Source | Licence | Attribution |
|---|---|---|---|
| Piano (`samples/piano/`): Kawai upright, notes every 3 semitones from A2 to C6, two velocity layers | **Upright Piano KW**, FreePats, version 2022-02-21, recorded by Gonzalo and Roberto. https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html | CC0 1.0 (the page and the bundled `cc0.txt` both say so) | Not required |
| Bass (`samples/bass/`): Yamaha RBX finger bass, E1 to D#2 | **Finger Bass YR**, FreePats, version 2019-09-30, by Andrea Biasior. https://freepats.zenvoid.org/ElectricGuitar/clean-electric-bass.html | CC0 1.0 (page and bundled `LICENSE.txt`) | Not required |
| Vibraphone (`samples/vibes/`): soft mallets, 11 notes from F3 to E6 | **Versilian Community Sample Library (VCSL)**, Vibraphone / Soft Mallets, by Sam Gossner. https://github.com/sgossner/VCSL | CC0 1.0 (the repository's LICENSE file) | Not required |
| Shaker (`samples/drums/shaker*`): four small-shaker hits | **VCSL**, Shaker, Small | CC0 1.0 | Not required |
| Nylon guitar (`samples/guitar/`): 9 notes from E3 to E5 | **Spanish Classical Guitar**, FreePats, version 2019-06-18, by Roberto. https://freepats.zenvoid.org/Guitar/acoustic-guitar.html | CC0 1.0 (page and bundled `cc0.txt`) | Not required |
| Rain (`samples/ambience/rain.m4a`): 40 s loop | **Rain against the window**, by cori, Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Rain_against_the_window.ogg | Public domain (Commons file page) | Not required |
| Café (`samples/ambience/cafe.m4a`): 40 s loop | **Restaurant ambience**, by stephan, Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Restaurant_ambience.ogg | Public domain (Commons file page) | Not required |
| Forest (`samples/ambience/forest.m4a`): 45 s loop of evening birdsong | **Waidachswald Oberschefflenz Abend 20250616 2005**, by Evillan, Wikimedia Commons. https://commons.wikimedia.org/wiki/File:Waidachswald_Oberschefflenz_Abend_20250616_2005.ogg | CC0 1.0 (Commons file page) | Not required |
| Stream (`samples/ambience/water.m4a`): 28 s loop | **363120 fractalstudios water trickle**, by FractalStudios, Wikimedia Commons. https://commons.wikimedia.org/wiki/File:363120_fractalstudios_water-trickle.wav | CC0 1.0 (Commons file page) | Not required |
| Drum hits (`samples/drums/`): kick, snare, rim and closed-hat one-shots | Sliced from a few loops of the **WaivOps Lo-Fi Drums Dataset** by Patchbanks / WaivOps, Zenodo, 2025, doi:10.5281/zenodo.15493892. https://zenodo.org/records/15493892 | CC BY 4.0 (Zenodo record metadata) | **Required.** The site's Credits popover credits it, and so does the block below. The hits are changed: cut from loops, faded and re-encoded |

## How the files were made

`engine-tools/build-samples.sh` trims the piano to 7 s with a 1.5 s fade and the bass to 3 s. It encodes everything as AAC (piano and drums at 96 kbps stereo, bass at 64 kbps mono). The drum one-shots were cut first from 18 streamed loops (out of 10,000) by onset detection and spectral class. Total size: about 1.6 MB.

The pad, the sub drones and round sub bass, the vinyl rumble, the build riser and a fallback rain loop are synthesised in `site/public/engine/render.js`, and the room bed in `site/public/engine/bed.js`. None of them uses recorded audio. The reversed piano swells are the piano samples played backwards. `engine-tools/build-samples-2.sh` builds the vibraphone, guitar and shaker files. `engine-tools/build-ambience.py` cuts the four ambience loops (2 s crossfade at the seam, -30 dBFS RMS, AAC, about 1.4 MB in all). A style fetches its ambience the first time it plays.

Two reference tracks were used only for spectral analysis while tuning the mix. None of their audio is in this repository or the site.

## CC BY 4.0 attribution (drum hits)

> **WaivOps Lo-Fi Drums Dataset** by Patchbanks / WaivOps, published on Zenodo (2025), doi:[10.5281/zenodo.15493892](https://doi.org/10.5281/zenodo.15493892), https://zenodo.org/records/15493892.
> Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
> **Changes made:** 18 loops were streamed from the archive; individual kick, snare, rim and closed-hat hits were cut out by onset detection, faded, level-matched and re-encoded as AAC (`engine-tools/slice-drums.py`, `engine-tools/build-samples.sh`). The original loops are not included.

## Public domain and CC0

The piano, bass, vibraphone, shaker, nylon guitar and the four ambience loops are CC0 1.0 or public domain. No attribution is required; the sources are credited above as a courtesy.

## Software

The site uses no third-party code. Training used open-source tools, listed in [DATA.md](DATA.md).
