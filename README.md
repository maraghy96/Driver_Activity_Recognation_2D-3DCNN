# Driver_Activity_Recognation_2D-3DCNN
Driver activity recognation using 2DCNN and 3DCNN
Using Drive&Act** Dataset (by Alina Roitberg and Manuel Martin) The task is to recognise the driver activity, for the intial benchmark there are 20 activities 'sitting_still': 0, 'eating': 1, 'fetching_an_object': 2, 'placing_an_object': 3,
            'reading_magazine':4, 'using_multimedia_display':5, 'talking_on_phone' : 6, 'writing': 7,
            'pressing_automation_button': 8, 'putting_on_jacket': 9, 'drinking': 10, 'fastening_seat_belt': 11,
            'taking_off_jacket': 12, 'looking_or_moving_around': 13, 'opening_bottle': 14, 'interacting_with_phone': 15,
            'working_on_laptop': 16,'reading_newspaper' : 17,
             'closing_bottle': 18, 'opening_laptop': 19.

For the intial commit, there are two versions of the code , the first is python notebook (.ipynb) and the second is .py files collection. 
The initial commit serves two model types namely 3DCNN and 2DCNN respectively.






**InProceedings{drive_and_act_2019_iccv,
author = {Martin, Manuel and Roitberg, Alina and Haurilet, Monica and Horne, Matthias and Reiß, Simon and Voit, Michael and Stiefelhagen, Rainer},
title = {Drive&Act: A Multi-modal Dataset for Fine-grained Driver Behavior Recognition in Autonomous Vehicles},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}



#For the main.py
Ensure you replace /path/to/data with the actual path to your dataset and adjust other parameters as needed.

Final Adjustments
Dataset and Model Paths: Adjust the paths in the script to match your directory structure.
Model and Training Details: Implement the train_model function with your training loop, handling any specifics of your training process.
Dependencies: Ensure all required libraries are installed in your environment, possibly using a virtual environment for isolation.
This setup allows you to run your model training from any environment with Python installed, moving beyond the constraints of a notebook-based environment like Colab.
