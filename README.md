# 🚗 Vehicle Trajectory Analysis and Visualization System
## Core Algorithm Details

![Version](https://img.shields.io/badge/Version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![License](https://img.shields.io/badge/License-MIT-orange)

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Vehicle+Trajectory+Analysis" alt="Vehicle Trajectory Analysis Diagram" width="700">
</p>

## 📑 Table of Contents

- [Introduction](#introduction)
- [Core Algorithms](#core-algorithms)
  - [1. Basic Distance and Similarity Metrics](#1-basic-distance-and-similarity-metrics)
  - [2. Trajectory Clustering Algorithms](#2-trajectory-clustering-algorithms)
  - [3. Spatio-temporal Optimization and Processing Techniques](#3-spatio-temporal-optimization-and-processing-techniques)
  - [4. Vehicle Motion Pattern Analysis Algorithms](#4-vehicle-motion-pattern-analysis-algorithms)
- [Project Structure and Execution](#5-project-structure-and-execution-brief)
- [Large Model Integration and Multimodal Analysis](#6-large-model-integration-and-multimodal-analysis)
- [Summary](#summary)

## Introduction

This project utilizes a series of computational geometry, machine learning, and spatio-temporal data analysis algorithms to perform in-depth processing, analysis, and visualization of vehicle GPS trajectory data. Its core lies in extracting valuable information from raw trajectory points, such as driving patterns, potential risks, and group behaviors. Below is a detailed explanation of the key algorithms used in the project.

## Core Algorithms

### 1. Basic Distance and Similarity Metrics

#### 1.1 Haversine Formula 🌐

*   **Principle**: The Haversine formula is used to calculate the great-circle distance (shortest distance) between two points on a sphere (like Earth). It uses the difference in latitude and longitude of the two points, combined with trigonometric functions (sine, cosine, arctangent `atan2`) and the Earth's radius \(R\), to compute the distance corresponding to the arc length. The formula accounts for the Earth's curvature, avoiding errors that might arise from using planar Euclidean distance over small areas, especially for longer distances or higher latitudes.
    
    ```math
    d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)}\right)
    ```
    
    Where \(\phi\) is latitude, \(\lambda\) is longitude (both must be converted to radians), and \(R\) is the Earth's radius (in meters in this project).

*   **Project Application**:
    * Calculating the distance between consecutive trajectory points for subsequent speed and acceleration calculations (`vehicleAnsys.py`) and displacement checks in turn analysis (`vehicleAnsys.py`, `bearDect.py`)
    * In collision detection, calculating the precise distance between spatially close vehicle pairs to determine if it's less than the collision threshold (`collisionDetect.py`)
    * Serving as the fundamental distance metric function for calculating the distance between two trajectory points in the DTW algorithm (`parse_vehicle_trajectories.py`, `dbscan_trajectory_clustering.py`)
    * Calculating the distance between trajectory centroids for rapid pre-filtering before DBSCAN clustering (`parse_vehicle_trajectories.py`, `dbscan_trajectory_clustering.py`)
    * Used to determine appropriate map extents and scale bars when generating visualization maps (`parse_vehicle_trajectories.py`, `vehicle_trajectory_video.py`)

#### 1.2 Dynamic Time Warping (DTW) ⏱️

*   **Principle**: DTW is an algorithm for measuring the similarity between two time series \(X=(x_1, ..., x_n)\) and \(Y=(y_1, ..., y_m)\), especially suitable when the series have different lengths or non-linear distortions in the time axis (like variations in speed). It constructs an \(n \times m\) cost matrix (where cell \((i, j)\) contains the distance between points \(x_i\) and \(y_j\), using Haversine distance in this project). It then finds a "Warping Path" from \((1, 1)\) to \((n, m)\) that represents the optimal alignment between the points of the two series, minimizing the cumulative cost. The warping path allows for one-to-many or many-to-one matching, handling temporal stretching and compression.

*   **Project Application**:
    * Serves as the core similarity (or distance) measure for trajectory clustering. By calculating the DTW distance between pairs of vehicle trajectories, their similarity in **shape** can be quantified, even if their travel speeds or sampling points differ
    * Implemented using the `fastdtw` library, an approximate DTW algorithm that significantly reduces the computational complexity of standard DTW (\(O(nm)\)) by calculating the path at a lower resolution and then refining it, making it feasible for longer trajectories
    * The DTW distances are used as input for subsequent DBSCAN or graph-based clustering algorithms

### 2. Trajectory Clustering Algorithms

#### 2.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 🔍

*   **Principle**: DBSCAN is a density-based clustering algorithm. It defines two core parameters: the neighborhood radius `eps` (\(\epsilon\)) and the minimum number of neighbors `min_samples` required to form a core point.
    1. **Core Point**: A point is a core point if its \(\epsilon\)-neighborhood contains at least `min_samples` points (including itself)
    2. **Border Point**: A point is a border point if it is not a core point but falls within the \(\epsilon\)-neighborhood of a core point
    3. **Noise Point**: A point that is neither a core nor a border point
    
    The algorithm starts with an arbitrary unvisited point and checks if it's a core point. If it is, a new cluster is created, and all points density-reachable (connected through a chain of core points) from this point are added to the cluster. If it's not a core point, it's temporarily marked as noise (it might later be found to be a border point of a cluster).

*   **Project Application**:
    * **Input**: A pre-computed \(N \times N\) distance matrix, where \(N\) is the number of vehicles to be clustered, and matrix element \((i, j)\) is the DTW distance between the trajectories of vehicle \(i\) and vehicle \(j\)
    * **Parameters**:
        * `eps`: Set by the `similarity_threshold` parameter (unit: meters). It defines how close the DTW distance must be to consider trajectories "similar" or "neighbors"
        * `min_samples`: The minimum cluster size. A valid trajectory cluster must contain at least `min_samples` trajectories that are sufficiently similar to each other (DTW distance < `eps`). The script dynamically estimates a suitable `min_samples` value based on the number of trajectories being clustered, avoiding the difficulty of manual setting
    * **Optimization**:
        * **Centroid Pre-filtering**: Since calculating DTW distances for all pairs of trajectories is very time-consuming (\(O(N^2)\) DTW computations), the script first calculates the geographical centroid (average latitude and longitude) of each trajectory. Then, DTW distance is only computed for pairs of vehicles whose centroids are closer than `centroid_threshold`. This significantly reduces the number of expensive DTW calculations needed
        * **Parallel Computation**: Uses `multiprocessing.Pool` to compute the DTW distances between the filtered pairs of vehicles in parallel
    * **Output**: A series of cluster lists, each containing the IDs of vehicles belonging to the same cluster. Trajectories identified as noise (not belonging to any sufficiently dense cluster) are initially labeled -1 by DBSCAN. The script subsequently treats these noise points, along with vehicles not included in the DBSCAN calculation (due to centroid filtering), as separate single-member clusters

*   **Why Chosen**: DBSCAN does not require specifying the number of clusters beforehand, can find arbitrarily shaped clusters (trajectory clusters are often irregular), and is robust to noise (outlier trajectories), making it well-suited for exploratory trajectory data mining.

#### 2.2 Graph-Based Connected Components Clustering 🔗

*   **Principle**: A simpler clustering approach.
    1. Construct a graph where each node represents a vehicle trajectory
    2. Add an undirected edge between two nodes if the DTW distance between their corresponding trajectories is less than a predefined `similarity_threshold`
    3. Find all connected components of this graph. Each connected component represents a cluster, meaning all trajectories within the component can be reached from each other via a series of similar trajectories (edges)

*   **Project Application**: (`parse_vehicle_trajectories_v1.py`)
    * Uses the NetworkX library to build the graph and find connected components
    * Can also be combined with centroid pre-filtering and parallel computation to optimize DTW calculations

*   **Why Chosen**: Relatively simple and intuitive to implement, suitable for finding tightly connected groups of trajectories. However, compared to DBSCAN, it might be less sensitive to density variations and could merge clusters connected by sparse "bridges".

### 3. Spatio-temporal Optimization and Processing Techniques

#### 3.1 KD-Tree 🌳

*   **Principle**: A KD-Tree is a binary tree data structure used to organize points in a K-dimensional space for efficient searching. When building the tree, it alternates between dimensions (longitude and latitude in this project), selecting a splitting point (often the median) to divide the current spatial region into two sub-regions, and recursively processes the sub-regions.

*   **Project Application**: (`collisionDetect.py`)
    * Within each time window (defined by time binning) for collision detection, a 2D KD-Tree is built using the vehicle position points (latitude, longitude) from that window and its adjacent windows
    * Utilizes the KD-Tree's `query_pairs` method to quickly find all pairs of points closer than `distance_threshold`. This avoids \(O(n^2)\) distance checks for all point pairs within the window, achieving an average complexity closer to \(O(n \log n)\)

#### 3.2 Time Binning ⏲️

*   **Principle**: Discretizes the continuous time dimension into a series of fixed-size time intervals (bins).

*   **Project Application**: (`collisionDetect.py`)
    * Assigns all trajectory points to different time bins based on their timestamps
    * Collision detection is primarily performed within the same time bin. To handle events crossing bin boundaries, points in adjacent time bins are typically also considered
    * This greatly narrows down the data range needed for each collision check, as only points close in time are likely to collide. The `time_bin_size` parameter controls the bin size

#### 3.3 Trajectory Downsampling and Interpolation 📉📈

*   **Downsampling**:
    * **Principle**: Selects a subset of points from the original trajectory sequence to reduce data volume. This project uses simple fixed-step sampling (e.g., `trajectory[::step]`, taking one point every `step-1` points)
    * **Project Application**: Downsamples trajectories before calculating DTW distances (`dbscan_trajectory_clustering.py`, `parse_vehicle_trajectories_v1.py`). This can significantly reduce DTW computation time (complexity depends on sequence length) but results in some loss of trajectory detail. The `step` parameter controls the sampling rate
    * **Why Chosen**: Simple to implement, very low computational overhead, effectively speeds up subsequent processing

*   **Interpolation**:
    * **Principle**: Generates new intermediate points between original trajectory points according to some rule (e.g., linear)
    * **Project Application**: Performs linear interpolation (`interpolate_trajectory` function) on original trajectories when generating trajectory animation videos (`vehicle_trajectory_video.py`). Inserts `factor-1` new points between two original points, making the point distribution denser and the animation appear smoother and more continuous

### 4. Vehicle Motion Pattern Analysis Algorithms

#### 4.1 Bearing Calculation 🧭

*   **Principle**: Calculates the direction angle (relative to true north) from point 1 (lat1, lon1) to point 2 (lat2, lon2). Typically uses the `atan2(y, x)` function, where x and y are projected coordinate differences calculated from latitude and longitude differences.
    
    ```math
    \theta = \operatorname{atan2}(\sin(\Delta\lambda) \cos(\phi_2), \cos(\phi_1) \sin(\phi_2) - \sin(\phi_1) \cos(\phi_2) \cos(\Delta\lambda))
    ```
    
    The result is usually converted to the 0-360 degree range.

*   **Project Application**: (`vehicleAnsys.py`, `bearDect.py`)
    * Calculates the driving direction for each consecutive segment of the trajectory
    * Forms the basis for subsequent turn analysis

#### 4.2 Circular Mean ⭕

*   **Principle**: Calculates the average of a set of angles, taking into account their periodic nature. The standard method involves converting each angle \(\alpha_i\) into a point on the unit circle \((\cos \alpha_i, \sin \alpha_i)\), calculating the centroid (average x-coordinate \(\bar{X}\) and average y-coordinate \(\bar{Y}\)) of these points, and then converting the centroid back to an average angle using `atan2(\bar{Y}, \bar{X})`.
    
    ```math
    \bar{\alpha} = \operatorname{atan2}\left(\frac{1}{N}\sum_{i=1}^N \sin \alpha_i, \frac{1}{N}\sum_{i=1}^N \cos \alpha_i\right)
    ```

*   **Project Application**: (`vehicleAnsys.py`, `bearDect.py`)
    * Calculates the average bearing of the initial few segments (`window_size`) of the trajectory (initial direction)
    * Calculates the average bearing of the final few segments of the trajectory (final direction)

#### 4.3 Speed, Acceleration, and Stop Detection 🚦

*   **Speed**: (`vehicleAnsys.py`)
    * Calculates the Haversine distance \(d_i\) between adjacent trajectory points \(i\) and \(i+1\), divided by the time difference between them \(\Delta t_i = t_{i+1} - t_i\). This gives the average speed \(v_i = d_i / \Delta t_i\) for segment \(i\)

*   **Acceleration**: (`vehicleAnsys.py`)
    * Estimated by calculating the rate of change of consecutive speeds. For example, the instantaneous acceleration at point \(i\) can be approximated as \((v_i - v_{i-1}) / ( (t_{i+1} - t_{i-1}) / 2 )\). Implementation requires careful handling of boundary conditions and time intervals

*   **Stop Detection**: (`vehicleAnsys.py`)
    * Identifies all segments where the speed \(v_i\) is below a `speed_threshold`
    * Merges consecutive low-speed segments that are continuous in time or separated by less than `merge_gap` into a single complete stop event. Records the start time, end time, and duration for each stop event

#### 4.4 Turn Analysis 🔄

*   **Principle**: (`vehicleAnsys.py`, `bearDect.py`)
    * Calculates the initial average direction \(\bar{\alpha}_{initial}\) and final average direction \(\bar{\alpha}_{final}\) of the trajectory (using circular mean)
    * Calculates the net turn angle \(\Delta \alpha = (\bar{\alpha}_{final} - \bar{\alpha}_{initial} + 180) \pmod{360} - 180\). The modulo operation ensures the result is within the \([-180, 180]\) degree range, where positive values typically indicate a left turn and negative values a right turn
    * Classifies the turn based on the magnitude of the net turn angle: U-Turn (close to \(\pm 180^\circ\)), Left Turn (e.g., \(> 70^\circ\)), Right Turn (e.g., \(< -70^\circ\)), Straight (close to \(0^\circ\)). Thresholds can be adjusted

### 5. Project Structure and Execution (Brief)

*   **Core Scripts**:
    * `sort_trajectory.py`: Sorts trajectory data by time
    * `collisionDetect.py`: Performs collision detection
    * `parse_vehicle_trajectories.py` / `dbscan_trajectory_clustering.py`: Trajectory clustering (DBSCAN)
    * `vehicleAnsys.py` / `bearDect.py`: Single trajectory motion/turn analysis
    * `vehicle_trajectory_video.py`: Generates trajectory animations

*   **Dependencies**: Major dependencies include `numpy`, `pandas`, `scipy`, `scikit-learn`, `fastdtw`, `matplotlib`, `networkx`, `contextily`, `tqdm`, `colorama`, etc. Install using `pip install -r requirements.txt` (if provided) or manually.

*   **Execution**: Typically run directly using `python <script_name.py>`. Some scripts may require modifying internal file paths or specifying input/output files via command-line arguments. Refer to comments within the scripts or use `--help` (if supported).

### 6. Large Model Integration and Multimodal Analysis

This project not only received assistance from Large Language Models (LLMs) during development but also integrates the capabilities of **multimodal large models** directly into the analysis workflow, enabling intelligent interpretation and report generation from visual results:

*   **Multimodal Input Processing**: Scripts `llm_png_anlyze.py` and `llm_video_analyze.py` leverage multimodal large models (like `qwen-vl-max`) capable of processing image/video and text information simultaneously. The model receives visual results of trajectories (cluster images or collision animations) along with related structured data (like vehicle speed, turn analysis results) as input.

    ```
    +-----------------------+       +-----------------------+
    | Trajectory Image (PNG)| ----> |                       |
    +-----------------------+       | Multimodal LLM        | ----> Formatted Report (HTML)
                                    | (e.g., Qwen-VL)       |
    +-----------------------+ ----> |                       |
    | Vehicle Data (JSON -> Text) | |                       |
    +-----------------------+       +-----------------------+
    (Workflow for llm_png_anlyze.py)

    +-----------------------+       +-----------------------+
    | Trajectory Video (MP4)| ----> |                       |
    +-----------------------+       | Multimodal LLM        | ----> Formatted Report (Markdown)
                                    | (e.g., Qwen-VL)       |
    +-----------------------+ ----> |                       |
    | Analysis Prompt (Text)|       |                       |
    +-----------------------+       +-----------------------+
    (Workflow for llm_video_analyze.py)
    ```

*   **Visual Trajectory Understanding**: Through carefully designed prompts, the LLM is guided to perform in-depth analysis of the input trajectory images or videos.
    * In `llm_png_anlyze.py`, the model is asked to identify main paths, turning patterns, and cluster characteristics in the image, and to infer driving behavior types by combining this with provided vehicle data (if available)
    * In `llm_video_analyze.py`, the model acts as a "traffic accident analysis expert," tasked with analyzing relative vehicle movements, speed changes, closest distances, assessing collision risks, and identifying key time points and evasive actions in the video

*   **Contextual Fusion Analysis**: The LLM is prompted to fuse visual information with additional textual data (vehicle analysis reports or cluster information from `output_results_2.json`) to make comprehensive judgments, generating analysis conclusions richer than those from a single data source.

    ```
       Visual Input     +     Textual Context     -->    LLM Fusion    -->   Enhanced Analysis
    (Image or Video)      (Vehicle Data, Prompt)          Engine              Conclusion
    ```

*   **Structured Report Generation**: Leverages the LLM's text generation capabilities to automatically create clearly formatted and content-rich analysis reports. The model is instructed to use specific formats (like HTML, Markdown), headings, lists, tables, and emphasis markers (like **bold**) to structure the output, enhancing the readability and professionalism of the reports.

*   **Automated Analysis Pipeline**: These scripts demonstrate the potential of integrating LLMs into data analysis pipelines, achieving end-to-end automation from raw data to visualization, natural language interpretation, and reporting.

This integration approach goes beyond simple code assistance, positioning the LLM as an intelligent component within the analysis process, responsible for understanding complex visual results and summarizing/evaluating them in a human-readable manner. This shows significant potential, especially for tasks requiring judgment based on a combination of visual patterns and data metrics.

## Summary

By cleverly combining and applying algorithms from various fields, this project achieves multi-dimensional, in-depth analysis and visualization of vehicle trajectory data, providing powerful tools for traffic management, behavioral analysis, and safety warnings. The choice of each algorithm is based on its suitability and efficiency in addressing specific problems, such as geographical distance calculation, sequence similarity, spatial indexing, density clustering, and kinematic analysis.

---

<p align="center">
  <b>Project Contributors</b><br>
  Development Team | Copyright © 2023
</p> 