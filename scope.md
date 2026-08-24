Project Objective:
    Inputs: 
        -> A video url from a public platform such as YouTube is given
        -> A text is given as input, ex: "My mind rebels at stagnation"
    Output:
        -> The timestamp of the identified frame
        -> The frame number, where applicable
        -> The extracted dialogue text
        -> The corresponding video frame as an image

High level Idea:
    -> Given the target the video, separate the video and audio part of the video. 
    -> Process the audio part to identify words with their corresponding duration and timeline.
    -> Match the extracted words with the target sentence
    -> If a valid match is found, match the audio and video component of the extracted sentence.
    -> Extract the video frames for the matched timeline only.

Non Functional Requirements:
    -> Ensure that the video processing does not take too long.
    -> Pipeline should be robust to normal variations in video quality, resolution, frame rate and etc.
    -> Pipeline should be generic and should support any video.
    -> Should handle similarity matching.
    -> Benchmarking different models.