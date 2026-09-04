# Per-frame ground truth for the dashcam video frames.
#
# cluster_manifest.json groups frames by adjacency and
# real_plate_transcriptions.json gives each cluster ONE hand-typed plate, which
# is then propagated to every frame in it. That assumption does not hold: the
# clusters were built from frame adjacency, and adjacency is not vehicle
# identity. Cluster 6 carries six different vehicles under the label
# "MH47Y1124"; cluster 39 carries eight under "MH01AH8495"; cluster 43 is
# thirteen frames of MH03CS6266 and one of the plate it is actually named
# after. Measured against the corrected labels, 37% of frames sat under a plate
# that is not the one in the picture - which capped per-frame accuracy far
# below what the reader was actually achieving.
#
# real_plate_frame_labels.json replaces that with one label per frame. It was
# built by grouping each cluster's frames by what the reader made of them,
# rendering a representative crop of every group, and reading the plate off the
# image by eye - the model proposes the grouping, a human assigns the string.
# 8 frames whose plate could not be read confidently are left out rather than
# guessed at.
