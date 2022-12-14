## ‘Labelling the Gaps’: A Weakly Supervised Automatic Eye Gaze Estimation

Over the past few years, there has been an increasing interest to interpret gaze direction in an unconstrained environment with limited supervision. Owing to data curation and annotation issues, replicating gaze estimation method to other platforms, such as unconstrained outdoor or AR/VR, might lead to significant drop in performance due to insufficient availability of accurately annotated data for model training. In this paper, we explore an interesting yet challenging problem of gaze estimation method with a limited amount of labelled data. The proposed method distills knowledge from the labelled subset with visual features; including identity-specific appearance, gaze trajectory consistency and motion features. Given a gaze trajectory, the method utilizes label information of only the start and the end frames of a gaze sequence. An extension of the proposed method further reduces the requirement of labelled frames to only the start frame with a minor drop in the generated label's quality. We evaluate the proposed method on four benchmark datasets (CAVE, TabletGaze, MPII and Gaze360) as well as web-crawled YouTube videos. Our proposed method reduces the annotation effort to as low as 2.67%, with minimal impact on performance; indicating the potential of our model enabling gaze estimation `in-the-wild' setup. [Paper](https://arxiv.org/pdf/2208.01840.pdf)

![Weakly supervised labelling approach illustration. The zig-zag path AD in the left is an example of a human gaze movement. This path can further be broken into several ‘gaze trajectories’ (AB,BC and CD). Given a gaze trajectory AB with gaze annotation for A and B, the objective of this work is to annotate the unlabelled frames i.e. P1, P2 and P3. The right side is an example of a gaze trajectory.](/figs/problem.png) 


## Dataset
Please download the data using the link: [Gaze360](https://github.com/erkil1452/gaze360/tree/master/dataset), [CAVE](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/), [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild), [TabletGaze](https://sh.rice.edu/cognitive-engagement/\%20tabletgaze/), [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation). 

## Pipeline 
![Overview of ‘2-labels’ and ‘1-label’ network pipelines. The frameworks take 3-frame set as input and learn to interpolate intermediate labels via weak supervision](/figs/one_two_labels.png) 

If you find the paper/code useful for your research, please consider citing our work:
```
@article{ghosh2022labelling,
  title={‘Labelling the Gaps’: A Weakly Supervised Automatic Eye Gaze Estimation},
  author={Ghosh, Shreya and Dhall, Abhinav and Hayat, Munawar and Knibbe, Jarrod},
  journal={arXiv preprint arXiv:2208.01840},
  year={2022}
}
```
