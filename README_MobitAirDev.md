### For enabling the 'allow debugging' on the Quest. 
```bash

# Connect Quest to robot computer with USB A or C 

adb devices 

# Put on Quest headset and accept the USB debugging popup that should have just appeared 
```

 

### Building the workspace 

# It needs a specific version of python for the Conda environment because of the ROS2 Jazzy requiring it 
```bash

conda create -n vt python=3.12 

Conda activate vt 

pip install empy==3.3.4 

conda install pinocchio==3.2.0 -c conda-forge 

pip install meshcat casadi pyyaml pure-python-adb 

# Install rest of packages 
```

 

### To run the code in robot computer 
```bash

# Plug Quest into robot computer 

adb devices 

# Accept debugging on Quest headset put on. 

conda activate vt 

source /opt/ros/jazzy/setup.bash 

cd ~/QuestArmTeleop 

source ./install/setup.bash 

sudo bash ~/QuestArmTeleop/src/agx_arm_ros/scripts/can_muti_activate.sh 

ros2 launch oculus_reader teleop_double_nero.launch.py 
```