# Orienting Polygonal Parts without Sensors: An Implementation in Python

This is a web browser implmentation of [Prof. Ken Goldberg's](https://goldberg.berkeley.edu/) PhD 
Thesis [Orienting Polygonal Parts Without Sensors (1993)](https://goldberg.berkeley.edu/pubs/algo93.pdf),
describes an algorithm for computing a series of squeeze actions that orient polygonal parts
*up to symmetry*. A Java implementation of the algorithm can be found [here](https://goldberg.berkeley.edu/part-feeder/),
and for more background on the algorithm can be found [here](https://goldberg.berkeley.edu/feeder/)
or by reading the thesis.

This was written in Python using the [Flask](https://flask.palletsprojects.com/) 
web framework, [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/) 
for interactive web plotting, and [pymunk](http://www.pymunk.org/en/latest/) to simulate the squeeze
actions for the animation.

## Description of files
The application is contained within the [part-feeder folder](part-feeder/). Inside that folder:
* [engine.py](part-feeder/engine.py) contains the functions related to the algorithm itself
* [feeder.py](part-feeder/feeder.py) uses the functions in engine.py to execute the algorithm and
generate the plots
* [anim.py](part-feeder/anim.py) contains the simulation of the squeeze actions using pymunk
* [explanations.py](part-feeder) contains the text seen on the webpage

The [dev folder](dev/) contains files used for development and experimental purposes, and are not 
related to the web application. Those can be safely ignored.
