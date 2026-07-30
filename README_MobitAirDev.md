// Connect Quest with USB C or USB A cable to robot computer.
// Login to terminal on robot computer through NoMachine, SSH, or natively.
abd devices
// Put on Quest and press 'Allow debugging' with hand controller button.
conda activate vt
source /opt/ros/jazzy/setup.bash
cd ~/QuestArmTeleop
source ./install/setup.bash
sudo bash ~/QuestArmTeleop/src/agx_arm_ros/scripts/can_muti_activate.sh
ros2 launch oculus_reader teleop_double_nero.launch.py