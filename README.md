# LeRobot Sim2Real


Replicate conda environment 
```
conda env create -f environment.yml

pip install -r requirements.txt
```

## Record demonstration and data 

To record a demonstration, run this script and then guide the robot to do a task: 
```
python lerobot_sim2real/scripts/robot_recorder.py
```
This will record the joint angles and end effector poses of the bot as well as save a video from the webcam. 

To annotate the frames of the recorded videos with bboxes from Gemini, run: 
```
python lerobot_sim2real/scripts/annotate_frames.py
```

To take the original frames, robot state data, bboxes from gemini and prepare one metadata file with only useful information for an LLM to view in its context, run: 
```
python lerobot_sim2real/scripts/prepare_metadata_and_frames.py
```
This will copy over the recorded frames, but will replace them with frames with bboxes if present. Since frames with bboxes are more useful for an LLM. 


