#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/ModelCoefficients.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "pcl/features/normal_3d.h"
#include "pcl/kdtree/kdtree.h"
#include "pcl/sample_consensus/sac_model_cylinder.h"
#include "pcl/sample_consensus/sac.h"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/voxel_grid.h>

// Measuring time
#include <chrono>

/// Type of the input point cloud contents
typedef pcl::PointXYZRGB PointT;
typedef pcl::RandomSampleConsensus<PointT> SampleConsensusEstimator;

class Shape
{
public:
  uint32_t id;
  Eigen::VectorXf _raw_coefficients;
  std::vector<int> inliers;
  struct
  {
    float inlier_proportion;
  } validity;
};

class Plane : public Shape
{
public:
  Eigen::Hyperplane<float, 3> model;

  void init(const uint32_t segment_id)
  {
    id = segment_id;
    model = Eigen::Hyperplane<float, 3>(_raw_coefficients.head<3>(), _raw_coefficients[3]);
  }
};

class Cylinder : public Shape
{
public:
  struct
  {
    struct
    {
      Eigen::Vector3f position;
      Eigen::Quaternion<double> orientation;
    } pose;
    float height;
    float radius;
  } model;
  struct
  {
    Eigen::ParametrizedLine<float, 3> axis;
    float radius;
  } pcl_model;

  void init(const uint32_t segment_id)
  {
    id = segment_id;
    pcl_model.axis = Eigen::ParametrizedLine<float, 3>(_raw_coefficients.head<3>(), _raw_coefficients.head<6>().tail<3>());
    model.radius = pcl_model.radius = _raw_coefficients[6];
  };
};

class CylinderEstimationNode : public rclcpp::Node
{
public:
  CylinderEstimationNode() : Node("ransac_cylinder")
  {
    // Declare parameters with default values
    this->declare_parameter<bool>("downsampling.enable", true);
    this->declare_parameter<double>("downsampling.voxel_size", 0.03);
    this->declare_parameter<double>("normal_est.normal_search_radius", 0.09);
    this->declare_parameter<int>("sac.min_points_for_cylinder", 50);
    this->declare_parameter<int>("sac.max_iterations", 1000);
    this->declare_parameter<double>("sac.distance_threshold", 0.15);
    this->declare_parameter<double>("sac.min_radius", 0.01);
    this->declare_parameter<double>("sac.max_radius", 0.5);
    this->declare_parameter<double>("sac.probability", 0.99);
    this->declare_parameter<double>("sac.normal_distance_weight", 0.1);
    this->declare_parameter<double>("result.inlier_proportion_threshold", 0.01);
    this->declare_parameter<bool>("result.textural_log_info", true);

    // Initialize subscriber to point cloud topic
    pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/zed2i/left/logs/log_pointcloud", 10,
        std::bind(&CylinderEstimationNode::pointcloudCallback, this, std::placeholders::_1));

    // Initialize publisher for marker array
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/ransac/log_cylinder_markers", 10);

    rclcpp::QoS qos_voxel_grid = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
    pub_voxel_grid_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ransac/log_pc_voxel_grid", qos_voxel_grid);
  }

private:
  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Fetch parameters
    bool downsample_input = this->get_parameter("downsampling.enable").as_bool();
    double voxel_leaf_size_xyz = this->get_parameter("downsampling.voxel_size").as_double();
    double normal_search_radius = this->get_parameter("normal_est.normal_search_radius").as_double();
    int min_points_for_cylinder_ = this->get_parameter("sac.min_points_for_cylinder").as_int();
    int param_max_iterations = this->get_parameter("sac.max_iterations").as_int();
    double param_distance_threshold = this->get_parameter("sac.distance_threshold").as_double();
    double param_min_radius = this->get_parameter("sac.min_radius").as_double();
    double param_max_radius = this->get_parameter("sac.max_radius").as_double();
    double probability = this->get_parameter("sac.probability").as_double();
    double param_normal_distance_weight = this->get_parameter("sac.normal_distance_weight").as_double();
    double inlier_proportion_threshold_ = this->get_parameter("result.inlier_proportion_threshold").as_double();
    bool textural_log_info = this->get_parameter("result.textural_log_info").as_bool();

    // Initialize marker array
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.clear();

    // Create a vector to store the detected cylinders
    std::vector<Cylinder> cylinders;

    // Convert ROS point cloud message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    // Downsample input to speed up the computations, if desired
    if (downsample_input)
    {
      RCLCPP_INFO(this->get_logger(), "Size before downsampling: %ld", pcl_cloud->size());
      pcl::VoxelGrid<PointT> voxel_grid;
      voxel_grid.setInputCloud(pcl_cloud);
      voxel_grid.setLeafSize(voxel_leaf_size_xyz,
                             voxel_leaf_size_xyz,
                             voxel_leaf_size_xyz);
      voxel_grid.filter(*pcl_cloud);
      RCLCPP_INFO(this->get_logger(), "Size after downsampling: %ld", pcl_cloud->size());
    }

    // Publish voxel grid
    sensor_msgs::msg::PointCloud2 msg_voxel_grid;
    pcl::toROSMsg(*pcl_cloud, msg_voxel_grid);
    msg_voxel_grid.header = msg->header;
    pub_voxel_grid_->publish(msg_voxel_grid);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    *working_cloud = *pcl_cloud;

    // Create a map to store the point clouds for each color
    std::map<std::tuple<uint8_t, uint8_t, uint8_t>, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> color_to_cloud_map;
    // Create new consideration of the geometric primitives
    Cylinder cylinder;
    // Each pointcloud cluster of already detected wood log is a different color, separate them
    for (const auto &point : working_cloud->points)
    {
      // Get the RGB color of the point
      std::tuple<uint8_t, uint8_t, uint8_t> color(point.r, point.g, point.b);

      // If the color is not already in the map, add it
      if (color_to_cloud_map.count(color) == 0)
      {
        color_to_cloud_map[color] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
      }

      // Add the point to the point cloud for its color
      color_to_cloud_map[color]->points.push_back(point);
    }

    int marker_id = 0;
    // For each color
    for (const auto &pair : color_to_cloud_map)
    {
      // Get the point cloud for the color
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud = pair.second;

      // Only proceed if the color cloud has enough points for a cylinder
      if (color_cloud->size() < (std::size_t)min_points_for_cylinder_)
      {
        RCLCPP_INFO(this->get_logger(), "Color cloud has less than %d points, skipping", min_points_for_cylinder_);
        break;
      }

      // Estimate normals for the segmented point cloud
      pcl::NormalEstimation<PointT, pcl::Normal> ne;
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

      pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

      ne.setSearchMethod(tree);
      ne.setInputCloud(color_cloud);
      ne.setKSearch(50);
      ne.compute(*cloud_normals);

      pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
      // Create the segmentation object for cylinder segmentation and set all the parameters
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_CYLINDER);
      /*The M-estimator sample consensus (MSAC), a generalization of the RANSAC estimator that is introduced by [20],
mainly used to robustly estimate multiple new relations from point correspondences. MSAC adopts the same sampling
strategy as RANSAC to generate reputed solutions but chooses the solution that maximizes the likelihood rather than
just the number of inliers. One of the problems in RANSAC is that if the threshold T for considering inlier is set too
high then the robust estimator can be very poor. Maximum likelihood estimation sample consensus (MLESAC) is a
case of MSAC that yields a modest to large benefit to all robust estimations with absolutely no additional
computational burden. The maximum likelihood error allows suggesting an improvement over MSAC. By locally
optimization this MSAC changed into LOMSAC.*/
      seg.setMethodType(pcl::SAC_MSAC);
      seg.setNormalDistanceWeight(param_normal_distance_weight);
      seg.setMaxIterations(param_max_iterations);
      seg.setDistanceThreshold(param_distance_threshold);
      seg.setRadiusLimits(param_min_radius, param_max_radius);
      seg.setInputCloud(color_cloud);
      seg.setInputNormals(cloud_normals);

      seg.segment(*inliers_cylinder, *coefficients_cylinder);

      // std::cout << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
      cylinder._raw_coefficients = Eigen::VectorXf::Map(coefficients_cylinder->values.data(), coefficients_cylinder->values.size());
      cylinder.inliers = inliers_cylinder->indices;

      // Get the inlier proportion
      float inlier_proportion = static_cast<float>((float)cylinder.inliers.size()) / static_cast<float>(working_cloud->size());

      // Check if the inlier proportion is above a certain threshold
      if (inlier_proportion < inlier_proportion_threshold_)
      {
        RCLCPP_INFO(this->get_logger(), "Inlier proportion is below threshold: %f", inlier_proportion);
        break;
      }

      // Init a cylinder
      cylinder.init(marker_id);
      // Extract cylinder limits from the associated pointcloud inliers
      // First create a hyperplane that intersects the cylinder, while being parallel with its flat surfaces
      Eigen::Hyperplane<float, 3> cylinder_plane(cylinder.pcl_model.axis.direction(), cylinder.pcl_model.axis.origin());

      // Iterate over all point cloud inliers and find the limits
      float min = 0.0, max = 0.0;
      RCLCPP_INFO(this->get_logger(), "Inliers: %ld", cylinder.inliers.size()); 
      for (auto &inlier_index : cylinder.inliers)
      {
        // Get signed distance to the point from hyperplane
        PointT point = color_cloud->at(inlier_index);
        Eigen::Vector3f point_position(point.x, point.y, point.z);
        float signed_distance = cylinder_plane.signedDistance(point_position);

        // Overwrite the limits if new are found
        double th = 4.0; // 4m maximum halflength
        if ((signed_distance < min) && (abs(signed_distance) < th)) 
        {
          min = signed_distance;
        }
        else if ((signed_distance > max) && (abs(signed_distance) < th))
        {
          max = signed_distance;
        }
      }
      // Print min and max points
      RCLCPP_INFO(this->get_logger(), "Min: %f, Max: %f", min, max);

      // Determine height of the cylinder
      cylinder.model.height = max - min;

      // Get centre of the cylinder and define it as the position
      cylinder.model.pose.position = (cylinder.pcl_model.axis.pointAt(min) + cylinder.pcl_model.axis.pointAt(max)) / 2.0;

      // Determine the orientation
      cylinder.model.pose.orientation.setFromTwoVectors(Eigen::Vector3d::UnitZ(), cylinder.pcl_model.axis.direction().cast<double>());

      // Push with the rest of the cylinders
      cylinders.push_back(cylinder);

      // Create a marker for the detected cylinder
      visualization_msgs::msg::Marker cylinder_marker;
      cylinder_marker.header.frame_id = msg->header.frame_id;
      // marker.header.stamp = msg->header.stamp; // could not get transform, so temporarily using time zero
      cylinder_marker.header.stamp = msg->header.stamp;//rclcpp::Time(0);
      cylinder_marker.id = marker_id++;
      cylinder_marker.type = visualization_msgs::msg::Marker::CYLINDER;
      cylinder_marker.action = visualization_msgs::msg::Marker::ADD;
      cylinder_marker.pose.position.x = cylinder.model.pose.position.x();
      cylinder_marker.pose.position.y = cylinder.model.pose.position.y();
      cylinder_marker.pose.position.z = cylinder.model.pose.position.z();
      cylinder_marker.pose.orientation.x = cylinder.model.pose.orientation.x();
      cylinder_marker.pose.orientation.y = cylinder.model.pose.orientation.y();
      cylinder_marker.pose.orientation.z = cylinder.model.pose.orientation.z();
      cylinder_marker.pose.orientation.w = cylinder.model.pose.orientation.w();
      cylinder_marker.scale.x = cylinder.model.radius * 2; // cylinder radius
      cylinder_marker.scale.y = cylinder.model.radius * 2; // cylinder radius
      cylinder_marker.scale.z = cylinder.model.height;     // cylinder height
      pcl::PointXYZRGB point = color_cloud->at(0);         // using the color coming from the cloud
      cylinder_marker.color.a = point.a / 255.0 / 2.0;     // half transparent
      cylinder_marker.color.r = point.r / 255.0;
      cylinder_marker.color.g = point.g / 255.0;
      cylinder_marker.color.b = point.b / 255.0;
      markers.markers.push_back(cylinder_marker);

      // Text Marker displaying Diameter, Length and Inlier Proportion of SAC fitted cylinder
      if (textural_log_info)
      {
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = msg->header.frame_id;
        text_marker.header.stamp = msg->header.stamp; //rclcpp::Time(0);
        text_marker.id = 1000 - marker_id; // TODO fix this
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position.x = cylinder.model.pose.position.x();
        text_marker.pose.position.y = cylinder.model.pose.position.y();
        text_marker.pose.position.z = cylinder.model.pose.position.z();
        text_marker.scale.z = 0.1;
        text_marker.color.a = 1.0;
        text_marker.color.r = point.r / 255.0;
        text_marker.color.g = point.g / 255.0;
        text_marker.color.b = point.b / 255.0;
        // Add the text to the marker
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2); // 2 digits after the decimal point
        // Add diameter, length and inlier proportion
        ss << "Diam.: " << 2 * cylinder.model.radius << "\nLen.: " << cylinder.model.height << "\nInlierProp.: " << inlier_proportion;
        text_marker.text = ss.str();
        markers.markers.push_back(text_marker);
      }
    }

    // Publish marker array
    marker_publisher_->publish(markers);
    auto end_time = std::chrono::high_resolution_clock::now();          // End time after the code
    std::chrono::duration<double> elapsed_time = end_time - start_time; // Calculate elapsed time

    std::cout << "Code took " << elapsed_time.count() << " seconds to execute.\n"; // Print the elapsed time
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_voxel_grid_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CylinderEstimationNode>());
  rclcpp::shutdown();
  return 0;
}
