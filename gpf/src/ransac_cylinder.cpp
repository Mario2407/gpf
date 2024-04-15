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
    // Initialize subscriber to point cloud topic
    pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/zed2i/left/logs/log_pointcloud", 10,
        std::bind(&CylinderEstimationNode::pointcloudCallback, this, std::placeholders::_1));

    // Initialize publisher for marker array
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/cylinder_markers", 10);

    // Declare parameters with default values
    this->declare_parameter<double>("normal_distance_weight", 0.1);
    this->declare_parameter<int>("max_iterations", 10000);
    this->declare_parameter<double>("distance_threshold", 0.15);
    this->declare_parameter<double>("min_radius", 0.01);
    this->declare_parameter<double>("max_radius", 0.5);
    this->declare_parameter<double>("normal_search_radius", 0.1);
    this->declare_parameter<double>("probability", 0.99);
  }

private:
  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Fetch parameters
    double normal_distance_weight = this->get_parameter("normal_distance_weight").as_double();
    int max_iterations = this->get_parameter("max_iterations").as_int();
    double distance_threshold = this->get_parameter("distance_threshold").as_double();
    double min_radius = this->get_parameter("min_radius").as_double();
    double max_radius = this->get_parameter("max_radius").as_double();
    double normal_search_radius = this->get_parameter("normal_search_radius").as_double();
    double probability = this->get_parameter("probability").as_double();

    // Convert ROS point cloud message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    *remaining_cloud = *pcl_cloud;

    // Initialize marker array
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.clear();

    std::vector<Cylinder> cylinders;

    // Create nullptr of sample consensus implementation
    SampleConsensusEstimator::Ptr sac;

    // Create nullptr of sample consensus models
    pcl::SampleConsensusModelCylinder<PointT, pcl::Normal>::Ptr sac_model_cylinder;

    // Create new consideration of the geometric primitives
    Cylinder cylinder;

    // Repeat RANSAC until no more cylinders are found

    // Create a map to store the point clouds for each color
    std::map<std::tuple<uint8_t, uint8_t, uint8_t>, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> color_to_cloud_map;

    // Each pointcloud cluster of already detected wood log is a different color, seperate them
    for (const auto &point : remaining_cloud->points)
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

    // Print the size of the color to cloud map
    // RCLCPP_INFO(this->get_logger(), "Color to cloud map size: %d", color_to_cloud_map.size());

    int marker_id = 0;
    // For each color
    for (const auto &pair : color_to_cloud_map)
    {
      // Get the point cloud for the color
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud = pair.second;

      // Only proceed if the color cloud has enough points for a cylinder
      if (color_cloud->size() > min_points_for_cylinder_)
      {
        // print color and according pointcloud size
        // RCLCPP_INFO(this->get_logger(), "Color: %d, %d, %d, Size: %d", std::get<0>(pair.first), std::get<1>(pair.first), std::get<2>(pair.first), color_cloud->size());

        // Estimate normals for the segmented point cloud
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(color_cloud);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZRGB>());
        normal_estimator.setSearchMethod(tree_n);
        normal_estimator.setRadiusSearch(normal_search_radius);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
        normal_estimator.compute(*cloud_normals);

        // Set the input cloud for the SAC model to the color cloud
        sac_model_cylinder.reset(new pcl::SampleConsensusModelCylinder<PointT, pcl::Normal>(color_cloud));
        sac_model_cylinder->setInputNormals(cloud_normals);
        sac_model_cylinder->setRadiusLimits(min_radius, max_radius);

        sac.reset(new SampleConsensusEstimator(sac_model_cylinder));
        sac->setDistanceThreshold(distance_threshold);
        sac->setMaxIterations(max_iterations);
        sac->setProbability(probability);

        if (!sac->computeModel())
        {
          RCLCPP_INFO(this->get_logger(), "Could not estimate a cylinder model for the given dataset of points: %d", color_cloud->size());
          // break;
        }

        // Get the cylinder coefficients
        sac->getModelCoefficients(cylinder._raw_coefficients);

        // Get the inliers
        sac->getInliers(cylinder.inliers);
        // RCLCPP_INFO(this->get_logger(), "Inliers: %d", cylinder.inliers.size());

        // Get the inlier proportion
        float inlier_proportion = static_cast<float>((float)cylinder.inliers.size()) / static_cast<float>(remaining_cloud->size());

        cylinder.init(marker_id);
        // Extract cylinder limits from the associated pointcloud inliers
        // First create a hyperplane that intersects the cylinder, while being parallel with its flat surfaces
        Eigen::Hyperplane<float, 3> cylinder_plane(cylinder.pcl_model.axis.direction(), cylinder.pcl_model.axis.origin());

        // Iterate over all point cloud inliers and find the limits
        float min = 0.0, max = 0.0;
        for (auto &inlier_index : cylinder.inliers)
        {
          // Get signed distance to the point from hyperplane
          PointT point = color_cloud->at(inlier_index);
          Eigen::Vector3f point_position(point.x, point.y, point.z);
          float signed_distance = cylinder_plane.signedDistance(point_position);

          // Overwrite the limits if new are found
          if (signed_distance < min)
          {
            min = signed_distance;
          }
          else if (signed_distance > max)
          {
            max = signed_distance;
          }
        }

        // Determine height of the cylinder
        cylinder.model.height = max - min;

        // Get centre of the cylinder and define it as the position
        cylinder.model.pose.position = (cylinder.pcl_model.axis.pointAt(min) + cylinder.pcl_model.axis.pointAt(max)) / 2.0;

        // Determne the orientation
        cylinder.model.pose.orientation.setFromTwoVectors(Eigen::Vector3d::UnitZ(), cylinder.pcl_model.axis.direction().cast<double>());

        // Push with the rest of the cylinders
        cylinders.push_back(cylinder);

        // Check if the inlier proportion is above a certain threshold
        // if (inlier_proportion < 0.5)
        // {
        //     RCLCPP_INFO(this->get_logger(), "Inlier proportion is below threshold.");
        //     break;
        // }

        // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Output the cylinder coefficients
        // if (inliers->indices.size() == 0)
        // {
        //   RCLCPP_INFO(this->get_logger(), "Could not estimate a cylinder model for the remaining dataset.");
        //   break;
        // }

        // // The seven coefficients of the cylinder are given by a point on its axis, the axis direction, and a radius, as:
        // // [point_on_axis.x point_on_axis.y point_on_axis.z axis_direction.x axis_direction.y axis_direction.z radius
        // Eigen::Vector3d axis_direction(cylinder._raw_coefficients[3], cylinder._raw_coefficients[4], cylinder._raw_coefficients[5]);

        // // Create a quaternion from the axis direction
        // Eigen::Quaterniond orientation = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axis_direction);

        // Create a marker for the detected cylinder
        visualization_msgs::msg::Marker cylinder_marker;
        cylinder_marker.header.frame_id = msg->header.frame_id;
        // marker.header.stamp = msg->header.stamp; // could not get transform, so temporarily using time zero
        cylinder_marker.header.stamp = rclcpp::Time(0);
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
        // Get the color from the point cloud
        pcl::PointXYZRGB point = color_cloud->at(0); // Assuming the color is the same for all points in the cloud
        cylinder_marker.color.a = point.a / 255.0 / 2.0; // Alpha value
        cylinder_marker.color.r = point.r / 255.0; // Red value
        cylinder_marker.color.g = point.g / 255.0; // Green value
        cylinder_marker.color.b = point.b / 255.0; // Blue value
        markers.markers.push_back(cylinder_marker);
      }
    }

    // visualization_msgs::msg::Marker cylinder_marker;
    // // Cylinders
    // for (auto &cylinder : cylinders)
    // {
    //   // geometric_primitive_msgs::msg::Cylinder c;
    //   cylinder_marker.id = cylinder.id;
    //   cylinder_marker.pose.position.x = cylinder.model.pose.position.x();
    //   cylinder_marker.pose.position.y = cylinder.model.pose.position.y();
    //   cylinder_marker.pose.position.z = cylinder.model.pose.position.z();
    //   cylinder_marker.pose.orientation.x = cylinder.model.pose.orientation.x();
    //   cylinder_marker.pose.orientation.y = cylinder.model.pose.orientation.y();
    //   cylinder_marker.pose.orientation.z = cylinder.model.pose.orientation.z();
    //   cylinder_marker.pose.orientation.w = cylinder.model.pose.orientation.w();
    //   // c.radius = cylinder.model.radius;
    //   // c.height = cylinder.model.height;
    //   // msg_primitives.primitives.cylinders.push_back(c);
    //   cylinder_marker.scale.x = cylinder.model.radius * 2; // cylinder radius
    //   cylinder_marker.scale.y = cylinder.model.radius * 2; // cylinder radius
    //   cylinder_marker.scale.z = cylinder.model.height;     // cylinder height
    //   cylinder_marker.color.a = 1.0;
    //   cylinder_marker.color.r = 0.0;
    //   cylinder_marker.color.g = 0.0;
    //   cylinder_marker.color.b = 1.0;
    //   markers.markers.push_back(cylinder_marker);
    // }
    // Publish marker array
    marker_publisher_->publish(markers);

    // while (remaining_cloud->size() > min_points_for_cylinder_)
    // {

    //   // Estimate normals for the segmented point cloud
    //   pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    //   normal_estimator.setInputCloud(remaining_cloud);
    //   pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZRGB>());
    //   normal_estimator.setSearchMethod(tree_n);
    //   normal_estimator.setRadiusSearch(normal_search_radius);
    //   pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    //   normal_estimator.compute(*cloud_normals);

    //   // save normals to file
    //   // pcl::io::savePCDFileASCII("normals.pcd", *cloud_normals);
    //   pcl::io::savePCDFileBinary("normals.pcd", *cloud_normals);

    //   sac_model_cylinder.reset(new pcl::SampleConsensusModelCylinder<PointT, pcl::Normal>(remaining_cloud));
    //   sac_model_cylinder->setInputNormals(cloud_normals);
    //   sac_model_cylinder->setRadiusLimits(min_radius, max_radius);

    //   sac.reset(new SampleConsensusEstimator(sac_model_cylinder));
    //   sac->setDistanceThreshold(distance_threshold);
    //   sac->setMaxIterations(max_iterations);
    //   sac->setProbability(0.99);

    //   sac->computeModel();
    //   // // Perform RANSAC
    //   // if (!sac->computeModel())
    //   // {
    //   //     RCLCPP_INFO(this->get_logger(), "Could not estimate a cylinder model for the given dataset.");
    //   //     break;
    //   // }

    //   // Get the cylinder coefficients
    //   // Eigen::VectorXf coefficients;
    //   sac->getModelCoefficients(cylinder._raw_coefficients);

    //   // Get the inliers
    //   // std::vector<int> inliers;
    //   sac->getInliers(cylinder.inliers);

    //   // Get the inlier proportion
    //   float inlier_proportion = static_cast<float>((float)cylinder.inliers.size()) / static_cast<float>(remaining_cloud->size());

    //   // Check if the inlier proportion is above a certain threshold
    //   // if (inlier_proportion < 0.5)
    //   // {
    //   //     RCLCPP_INFO(this->get_logger(), "Inlier proportion is below threshold.");
    //   //     break;
    //   // }

    //   pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    //   // // Output the cylinder coefficients
    //   // if (inliers->indices.size() == 0)
    //   // {
    //   //     RCLCPP_INFO(this->get_logger(), "Could not estimate a cylinder model for the remaining dataset.");
    //   //     break;
    //   // }

    //   // Create a marker for the detected cylinder
    //   visualization_msgs::msg::Marker marker;
    //   marker.header.frame_id = msg->header.frame_id;
    //   marker.header.stamp = msg->header.stamp;
    //   marker.type = visualization_msgs::msg::Marker::CYLINDER;
    //   marker.action = visualization_msgs::msg::Marker::ADD;
    //   marker.pose.position.x = cylinder._raw_coefficients[0];
    //   marker.pose.position.y = cylinder._raw_coefficients[1];
    //   marker.pose.position.z = cylinder._raw_coefficients[2];
    //   marker.pose.orientation.x = 0.0;
    //   marker.pose.orientation.y = 0.0;
    //   marker.pose.orientation.z = 0.0;
    //   marker.pose.orientation.w = 1.0;
    //   marker.scale.x = cylinder._raw_coefficients[6] * 2; // cylinder radius
    //   marker.scale.y = cylinder._raw_coefficients[6] * 2; // cylinder radius
    //   marker.scale.z = cylinder._raw_coefficients[5];     // cylinder height
    //   marker.color.a = 1.0;
    //   marker.color.r = 0.0;
    //   marker.color.g = 0.0;
    //   marker.color.b = 1.0;
    //   markers.markers.push_back(marker);

    //   // Extract the inliers of the cylinder and remove them from the remaining cloud
    //   pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    //   extract.setInputCloud(remaining_cloud);
    //   extract.setIndices(inliers);
    //   extract.setNegative(true);
    //   extract.filter(*remaining_cloud);
    // }

    // // Publish marker array
    // marker_publisher_->publish(markers);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  size_t min_points_for_cylinder_ = 50; // Minimum number of points required to detect a cylinder: 100?
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CylinderEstimationNode>());
  rclcpp::shutdown();
  return 0;
}
