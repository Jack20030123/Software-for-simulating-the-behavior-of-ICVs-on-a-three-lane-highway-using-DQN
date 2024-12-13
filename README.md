# Software-for-simulating-the-behavior-of-ICVs-on-a-three-lane-highway-using-DQN
This software, based on the DQN model, is primarily designed to study the lane-changing decision mechanisms of ICVs. It provides a detailed simulation environment tailored to various real-world road conditions, including different traffic densities, numbers of ICVs, and specific ICV parameter settings.

1 Operating Environment

The software can be run on most mainstream commercial computers equipped with 64-bit Windows operating systems, macOS, or Linux. The lane-changing decision-making software for Intelligent Connected Vehicles (ICVs) requires Python 3.6 or above, with additional libraries such as tkinter, PyTorch, numpy, matplotlib, pandas, and highway_env.

Additionally, the computer must have Microsoft Office or WPS software installed to facilitate the output and processing of simulation data.

2 User Guide

2.1 Software Execution
Open the "start.py" file in Python and run it to display the software menu interface, as shown in Figure 1.

<div align="center">
  <img src="https://github.com/user-attachments/assets/cad4bfdd-f918-49e2-9a8f-21989ff8b8cb" alt="Image Description" width="200">
</div>
<p align="center">
  Figure 1: Software Menu Interface
</p>

2.2 Execution Steps

2.2.1 Parameter Configuration

After successfully entering the number of ICVs and HDVs, the parameter configuration interface will appear, as shown in Figure 2. In this interface, users can adjust the reward functions and simulation settings for each ICV. The number of vehicle parameter input fields will match the number of ICVs entered in the previous interface. After completing the input, click "Submit" to begin training.

<div align="center">
  <img src="https://github.com/user-attachments/assets/29c60ee3-315e-4231-82ab-c746078b115e" alt="Image Description" width="200">
</div>

<p align="center">
  Figure 2: Parameter Configuration Interface
</p>

2.2.4 Simulation Execution

Upon clicking "Submit," the software will automatically start model training, as shown in Figure 3. The software will initiate highway-env to conduct model training and evaluation in a three-lane highway scenario. Green vehicles represent ICVs, while blue vehicles represent HDVs. The ICVs can achieve coordinated control among themselves.

<div align="center">
  <img src="https://github.com/user-attachments/assets/a7d1a872-6feb-4445-8fe7-e437b37b3f66" width="800">
</div>

<p align="center">
  Figure 3: Simulation Interface
</p>

2.2.5 Simulation Results Display

Once the simulation training is complete, evaluation results in the form of images and Excel files will be stored in the local train folder and can be directly viewed in PyCharm. Key results include the ICVs' average speed line chart, reward value function graph, survival time, and headway distance.

<div align="center">
  <img src="https://github.com/user-attachments/assets/ffb532a4-c9e2-4415-ad32-1d252864b2e0" width="800">
</div>

<p align="center">
  Figure 4: Average Reward Per Vehicle During Each Training Session
</p>
