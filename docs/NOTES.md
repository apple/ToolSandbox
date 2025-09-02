# Running notes for my reference

Goal Inference tasks

1. connectivity_check_location_share
2. coordinate_ride_with_distance
3. dentist_appointment_location_services
4. dinner_plan_review_and_confirmation
5. dinner_plans_charlie_reminder
6. doctor_appointment_location_family_notify
7. doctor_appointment_timestamp_emergency_contact
8. friend_arrival_contact_and_reminder
9. long_drive_preparation_multiple_settings
10. office_meeting_distance_timed_reminder



We have implemented a trajectory parser @.claude/tasks/TRAJECTORY_PARSER.md. Now we
want to use this and LiteLLMmodel class to run inference on different models to see
if the models can perform goal inference.\
The workflow goes like this, parse the trajectory.\
1. Prepare a system prompt to be used for goal inference task.\
2. Each user message to the model should be one tool call and the resulting database
changes. \
3. And at each step, the agent should either predict the goal or output wait. Here
is where we need to use the @tool_sandbox/models/utils.py ParseFunction class and
define an output parser for the model. We want a simple boxed parser. The model
should either output [[wait]] or [[inferred goal]] and nothing else. If there is a
format error, we should reprompt the model showing the model output and the correct
format. See ParseFunction to see how parser should be implemented. You can also take
a look at my parser implementation for a different project here:
https://github.com/deepakn97/abductive-dataset/blob/main/par/hypgen/parsers.py\
4. We should then store the output of the model at each step. \
\
Couple of tips:\
1. You need to provide explanation of what each database is in the system prompt.\
2. You also need to provide an example of a trajectory in the system prompt.\
3. Use existing LiteLLM model class to set up the inference.\


Plan the changes very thoroughly. And understand the codebase thoroughly before planning the changes.
