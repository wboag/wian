# notevec
AMIA submission examining note-derived vector representations


This work proposes to investigate note representations from three perspectives:

    1. *What information is captured in the notes?* mimic has so many structured variables that we can try predicting: length of stay, first_careunit, last_careunit, first_wardid, last_wardid, admission_type, admission_location, insurance, language, religion, marital_status, diagnosis, etc. We would expect that some of these are in the notes (though not all, obviously, and we can filter the impossible-to-predict tasks out), but for the ones that are, we can measure how well various note representations are able to encode that information. We do this with a suite of simple prediction tasks.

    2. *Do similar notes have similar representations?* This is easy to advocate for from a precision medicine perspective, but it is also interesting from a purely modeling point of view as well. We can run qualitative knn searches as well as possibly clustering to measure the representation space, itself. I love a juicy qualitative analysis in a paper that let's the reading get a sense for what's actually going on. No one likes a paper that is just tables and tables of numbers.

    3. *Do these representations have predictive power for clinical tasks?* Can the note-derived representations predict clinically relevant outcomes, such as in-hospital?

