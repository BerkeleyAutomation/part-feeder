import dash_core_components as dcc

ch_ap = dcc.Markdown('''
## The Polygon

The polygon you entered is shaded in blue. The red outline represents the polygon's convex hull,
which the algorithm uses to compute a plan. The black lines that cross the polygon are antipodal
pairs. Antipodal pairs are pairs of points that admit parallel supporting lines, which are used
to compute the diameter function. For more on antipodal pairs, see Preparata and Shamos (1985).

The convex hull is computed using Jarvis's March, which runs in O(nh) time. The antipodal pairs
algorithm runs in O(n) time. 
''')

dia = dcc.Markdown('''
## The Diameter Function

The diameter function represents the minimum distance between two parallel lines as they are rotated
around the polygon. This is computed using an adapted version of the antipodal pairs algorithm. The
diameter function is a piecewise function consisting of a series of sinosoids whose amplitude and 
phase shift vary with respect to the polygon. For more info on computing the diameter function, see
Goldberg (1993). 

The red vertical lines represent the local maxima of the function, while the blue lines represent 
the local minima. These extrema are used to compute the squeeze function, shown below.
''')

sq = dcc.Markdown('''
## The Squeeze Function

The squeeze function is a transfer function that represents the output orientation of the part with
respect to the gripper if the polygon was squeezed at a certain angle. This is computed by mapping
ranges between local maxima to the local minimum in between. For more info on the squeeze function, 
see Goldberg (1993). 

The discontinuities in the squeeze function define s-intervals and s-images, which are used to
compute the plan. 
''')

rad = dcc.Markdown('''
## The Radius Function

The radius function describes how the distance between a support line and the centroid of the 
polygon varies as the support line is rotated around the part. As with before, the red lines 
represent local maxima and blue lines represent local minima. Similar to the diameter function,
the extrema of the radius function are used to compute another transfer function: the push function.
''')

pu = dcc.Markdown('''
## The Push Function

The push function, similar to the squeeze function, is a transfer function that maps an initial
orientation of a part of the final orientation after a part is pushed by a single gripper jaw for
a sufficient distance such that one of the polygon's edges is aligned with the gripper.
The push function is derived in the same fashion as the squeeze function is derived from the diameter
function. For more information on the push and radius function, see Goldberg (1993).

The push function is used to compute the push-grasp function, another transfer function that is used
to compute a different, more realistic plan.
''')

pg = dcc.Markdown('''
## The Push-Grasp Function

The push-grasp function is computed by composing the push function with the squeeze function. This
function (more specifically, its discontinuities) are used to generate a plan to orient the part
up to symmetry.
''')
