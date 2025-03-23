-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Dec 11, 2021 at 08:19 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `face_door_open`
--

-- --------------------------------------------------------

--
-- Table structure for table `fd_face`
--

CREATE TABLE `fd_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `fd_face`
--

INSERT INTO `fd_face` (`id`, `vid`, `vface`) VALUES
(1, 1, '1_2.jpg'),
(2, 1, '1_3.jpg'),
(3, 1, '1_4.jpg'),
(4, 1, '1_5.jpg'),
(5, 1, '1_6.jpg'),
(6, 1, '1_7.jpg'),
(7, 1, '1_8.jpg'),
(8, 1, '1_9.jpg'),
(9, 1, '1_10.jpg'),
(10, 1, '1_11.jpg'),
(11, 1, '1_12.jpg'),
(12, 1, '1_13.jpg'),
(13, 1, '1_14.jpg'),
(14, 1, '1_15.jpg'),
(15, 2, '2_2.jpg'),
(16, 2, '2_3.jpg'),
(17, 2, '2_4.jpg'),
(18, 2, '2_5.jpg'),
(19, 2, '2_6.jpg'),
(20, 2, '2_7.jpg'),
(21, 2, '2_8.jpg'),
(22, 2, '2_9.jpg'),
(23, 3, '3_2.jpg'),
(24, 3, '3_3.jpg'),
(25, 3, '3_4.jpg'),
(26, 3, '3_5.jpg'),
(27, 3, '3_6.jpg'),
(28, 3, '3_7.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `fd_history`
--

CREATE TABLE `fd_history` (
  `id` int(11) NOT NULL,
  `rid` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `vface` varchar(20) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `fd_history`
--

INSERT INTO `fd_history` (`id`, `rid`, `vid`, `name`, `vface`, `dtime`) VALUES
(1, 1, 1, 'Vijay', '600.jpg', '2021-12-11 00:26:34'),
(2, 1, 2, 'Santhosh', '375.jpg', '2021-12-11 00:27:05'),
(3, 1, 1, 'Vijay', '278.jpg', '2021-12-11 00:28:14'),
(4, 1, 1, 'Vijay', '598.jpg', '2021-12-11 00:28:47'),
(5, 1, 0, 'Unknown', '313.jpg', '2021-12-11 00:30:42'),
(6, 1, 1, 'Vijay', '314.jpg', '2021-12-11 00:44:58'),
(7, 1, 1, 'Vijay', '429.jpg', '2021-12-11 00:50:14'),
(8, 1, 0, 'Unknown', '123.jpg', '2021-12-11 00:50:42'),
(9, 1, 1, 'Vijay', '325.jpg', '2021-12-11 00:51:52'),
(10, 1, 0, 'Unknown', '882.jpg', '2021-12-11 00:52:26'),
(11, 1, 0, 'Unknown', '812.jpg', '2021-12-11 07:10:20'),
(12, 1, 0, 'Unknown', '299.jpg', '2021-12-11 08:00:32'),
(13, 1, 0, 'Unknown', '774.jpg', '2021-12-11 08:01:51'),
(14, 1, 0, 'Unknown', '455.jpg', '2021-12-11 08:04:43'),
(15, 1, 1, 'Vijay', '803.jpg', '2021-12-11 08:58:02'),
(16, 1, 1, 'Vijay', '270.jpg', '2021-12-11 08:58:25'),
(17, 1, 0, 'Unknown', '334.jpg', '2021-12-11 08:58:54'),
(18, 1, 0, 'Unknown', '971.jpg', '2021-12-11 10:32:59'),
(19, 1, 1, 'Vijay', '875.jpg', '2021-12-11 12:03:36'),
(20, 1, 1, 'Vijay', '537.jpg', '2021-12-11 12:04:22'),
(21, 1, 1, 'Vijay', '613.jpg', '2021-12-11 12:05:35'),
(22, 1, 1, 'Vijay', '713.jpg', '2021-12-11 12:05:42'),
(23, 1, 1, 'Vijay', '314.jpg', '2021-12-11 12:13:03'),
(24, 1, 0, 'Unknown', '370.jpg', '2021-12-11 12:33:45'),
(25, 1, 0, 'Unknown', '207.jpg', '2021-12-11 12:36:16'),
(26, 1, 0, 'Unknown', '586.jpg', '2021-12-11 12:38:33'),
(27, 1, 0, 'Unknown', '549.jpg', '2021-12-11 12:40:40'),
(28, 1, 0, 'Unknown', '862.jpg', '2021-12-11 12:42:29'),
(29, 1, 0, 'Unknown', '245.jpg', '2021-12-11 12:44:44'),
(30, 1, 0, 'Unknown', '782.jpg', '2021-12-11 12:49:27'),
(31, 1, 0, 'Unknown', '753.jpg', '2021-12-11 12:50:23'),
(32, 1, 0, 'Unknown', '339.jpg', '2021-12-11 12:51:36'),
(33, 1, 0, 'Unknown', '855.jpg', '2021-12-11 12:54:16'),
(34, 1, 0, 'Unknown', '441.jpg', '2021-12-11 12:54:48'),
(35, 1, 2, 'Santhosh', '810.jpg', '2021-12-11 12:55:20'),
(36, 1, 0, 'Unknown', '301.jpg', '2021-12-11 12:55:43'),
(37, 1, 1, 'Vijay', '373.jpg', '2021-12-11 12:56:12'),
(38, 1, 0, 'Unknown', '713.jpg', '2021-12-11 12:56:33'),
(39, 1, 0, 'Unknown', '670.jpg', '2021-12-11 12:59:57'),
(40, 1, 0, 'Unknown', '713.jpg', '2021-12-11 13:01:30'),
(41, 1, 0, 'Unknown', '494.jpg', '2021-12-11 13:02:38'),
(42, 1, 0, 'Unknown', '610.jpg', '2021-12-11 13:04:24'),
(43, 1, 0, 'Unknown', '238.jpg', '2021-12-11 13:09:26'),
(44, 1, 0, 'Unknown', '469.jpg', '2021-12-11 13:12:01'),
(45, 1, 0, 'Unknown', '370.jpg', '2021-12-11 13:19:14'),
(46, 1, 0, 'Unknown', '380.jpg', '2021-12-11 13:19:58'),
(47, 1, 0, 'Unknown', '601.jpg', '2021-12-11 13:25:55'),
(48, 1, 0, 'Unknown', '282.jpg', '2021-12-11 13:31:33'),
(49, 1, 0, 'Unknown', '302.jpg', '2021-12-11 13:32:58'),
(50, 1, 1, 'Vijay', '993.jpg', '2021-12-11 13:39:12'),
(51, 1, 1, 'Vijay', '281.jpg', '2021-12-11 13:40:28');

-- --------------------------------------------------------

--
-- Table structure for table `fd_register`
--

CREATE TABLE `fd_register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `detail` varchar(50) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `fimg` varchar(30) NOT NULL,
  `rid` int(11) NOT NULL,
  `utype` varchar(20) NOT NULL,
  `detect` int(11) NOT NULL,
  `vface` varchar(20) NOT NULL,
  `pin` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `fd_register`
--

INSERT INTO `fd_register` (`id`, `name`, `mobile`, `detail`, `email`, `uname`, `pass`, `rdate`, `fimg`, `rid`, `utype`, `detect`, `vface`, `pin`) VALUES
(1, 'Vijay', 9976570006, '', 'viji@gmail.com', 'vijay', '1234', '10-12-2021', '1_15.jpg', 1, 'admin', 0, '', '3809'),
(2, 'Santhosh', 8940228614, 'Brother', '', 'u2', '1234', '10-12-2021', '2_9.jpg', 1, 'user', 0, '', ''),
(3, 'Raj', 9003938949, 'Brother', '', 'u3', '1234', '11-12-2021', '', 1, 'user', 0, '', '');

-- --------------------------------------------------------

--
-- Table structure for table `fd_temp`
--

CREATE TABLE `fd_temp` (
  `id` int(11) NOT NULL,
  `vface` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `fd_temp`
--

INSERT INTO `fd_temp` (`id`, `vface`) VALUES
(1, '4_2.jpg'),
(2, '4_3.jpg'),
(3, '4_4.jpg'),
(4, '4_5.jpg'),
(5, '4_6.jpg'),
(6, '4_7.jpg');
