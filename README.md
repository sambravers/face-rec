# This is a face recognizer, or hopefully will be soon

The end goal of this project is to develop a neural network that doesn't have to accurately distinguish 100% between acceptable and rejectable face, but instead has a very low false-positive rate. I hope to eventually use this to drive a real-world door lock (i.e. lots more work), which would take a video and then only unlock above a certain (below 100%) threshold, given an acceptably low false-positive rate.

Included are .npy files that represent a small portion of the data that is accessible for further development of the model. However, that data is too large, so the reduced sized sets are included; this also, thankfully, reduces the class imbalance of the model.

I'm currently using the aligned images datasets (or rather, a random subset of that data) from the larger [YTF dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/index.html), compiled from YouTube videos by Tel Aviv university et al.

Apologies for the lack of full history in this repo - I realized, after already having spent considerable time building the tool, that it was unwise to release what I had just proven was a sufficient dataset of myself for facial recognition. However, there's still much work to-be-done and the rest should be tracked, here.

### Concept:

This tool is nothing novel; I essentially just took a dataset large enough that it clearly defines a general "face" as the reject bin and any user that I wished to accept as the accept bin. For the purpose's of this public-facing repo, what used to be my face has been replaced by Alfonso Cuaron's, as the developmental accept bin. Anyone wishing to use this tool for their own personal use can simply refill /alfonso\_cuaron with properly formatted images of their own face and retrain/save the network.

### TODO:

I will almost certainly be moving from one, front-facing image to pairs of slightly off-axis images. It would be ideal to eventually implement this as a non-connected device (e.g. a set of cameras connected to a controller), for security's sake. As a result, a front-facing image is insufficient, since the network could be fooled by a flat, paper picture of an accepted user.

To combat this, I need to either find or collect a dataset containing faces from two, consistent pan angles. I hope that the end-implementation of this will be (essentially) as follows: an array of cameras and a button that are mounted outside the physical door, which connect to a Raspberry Pi that, upon the button being pressed, will capture a certain number of frames from the cameras, run them through the final classifier network, and then operate a winch to turn the door handle from the inside, if the user is accepted.
