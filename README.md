### This is a face recognizer, or hopefully will be soon

The end goal of this project is to develop a neural network that doesn't have to accurately distinguish 100% between acceptable and rejectable face, but instead has a very low false-positive rate. I hope to eventually use this to drive a real-world door lock (i.e. lots more work), which would take a video and then only unlock above a certain (below 100%) threshold, given an acceptably low false-positive rate.

Included are .npy files that represent a small portion of the data that is accessible for further development of the model. However, that data is too large, so the reduced sized sets are included; this also thankfully lowers the class imbalance of the model.

I'm currently using the aligned images datasets from the larger [YTF dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/index.html), compiled from YouTube videos by Tel Aviv university et al.

## TODO:

Almost certainly will be moving from one, front-facing image to pairs of slightly off-axis images. It would be ideal to eventually implement this as a non-connected device (e.g. a set of cameras connected to a Raspberry Pi), for security's sake. As a result, a front-facing image is insufficient, since the network could be fooled by a flat, paper picture of an accepted user.

To combat this, I need to either find or collect a dataset containing faces from two, consistent pan angles. I also need to check out automatic, consistent face-to-frame proportion face-finders, as those would also make my job easier.
