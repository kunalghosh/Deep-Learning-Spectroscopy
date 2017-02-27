# Usage
* create a new folder, this represents a new experiment
* write a script which does the following:
** clones this repository
** checks out a specific revision (This represents a revision of the code which is used to run the experiment, since its in the repo, the revision identifies the exact code used to run the experiment)
** invokes the python script with the necessary arguments, this ensures the arguments used for the experiment are saved.

_NOTE:_ At anytime, there would be only one experiment run inside a folder so the above steps are fine. If you need to execute a different revision of another script in this repo (e.g. vis.py) then
you would need to clone the repo again and then revert back to the specific revision to use the script. This is clumsy, but untill a better way is figured out, this would be our approach.