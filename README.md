# ocean_simulator

Fast Finite Volume Python implementation of scalar transport in the ocean with GUI visualisation of concentration fields.

![ocean_simulator](https://github.com/user-attachments/assets/045fb4b9-b196-4856-aaa7-c5746545535f)

The code is a fast implementation of Finite Volume Method for the Advection Diffusion equation in 2D uniform grids to simulate surface transport in the ocean. Velocity fields were taken from Mars3D simulation (Ifremer) of the year 2023.  Advection is solved with a 2nd order flux-limited Lax-Wendroff scheme and diffusion is solved in the frequency domain, with buffer cells for boundary conditions.

The code solve and plot the transported fields in real time, with about 60 frames per seconds on a average laptop.

User interaction with the scalar field is possible via left and right mouse click.

To run : 
- Download at least 1 month of 2023 velocity currents data at https://doi.org/10.57745/RJS7KV (about 600mb per month). If only one month is taken, it should be the first one (January)
- Download the python code ocean_simulator_GUI.py
- (optional) Download the tide level file 

All data needs to be place in the same folder as the code.

Run with :

- python3 ocean_simulator_GUI.py

Ii needed, install missing packages with pip :

- pip install scipy glob argparse pyqt5 h5py multiprocessing

Enjoy and acknowledge : Joris Heyman, Geosciences Rennes, Univ Rennes, CNRS. (joris.heyman@univ-rennes)
