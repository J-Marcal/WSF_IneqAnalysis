# Assessing inequalities in urban water security through geospatial analysis

This repository provides the code for the analysis carried out to support the results presented in the paper : *Assessing inequalities in urban water security through geospatial analysis*, submitted to the Safe and Sustainable Water in Cities  special issue of PLOS WATER. Please consider citing the paper if you find the paper or the code useful for you.

## About the project

The project aims to evaluate urban water security trough a geospatial perspective. The goal is to investigate inequalities by analysing water security at intra-urban scale. This code was written to support the following analysis: spatial interpolation, computation of the Theil entropy index, map plotting, results visualisation and spatial correlation. 

The assessment of water security for which all the analysis is carried out is presented in the paper and is based in a 4 dimension framework that included aspects on *A:  Drinking water and human well-being*, *B: Ecosystems*, *C: Water related hazards and climate change* and *D: Economic and social development*.  The framework has a hierarchical organization, with dimensions divided into categories and those into indicators. For each step down on the hierarchical organization, a number is attributed to the category and then to the indicator (for example, dimension A, category A1, indicator A1.1). 

The files correspond to the application of the framework to the city of Campinas in Brazil. 

## Requirements

### Programming language and packages
The code is written in Python, therefore, to execute it you first need to install Python on your system. Version 3.8.6 was the one used  in the development of this code. 

The packages that need to be installed and the versions used in the development are listed below:
|  Python library|Version|
|----------------|------------|
|pandas|1.1.4          
|geopandas |0.10.2
|numpy|1.21.6
|matplotlib|3.3.3
|esda|2.4.1
|scikit-learn |1.1.1
|seaborn| 0.11.2
|pysal   |  2.6.0                                         

### Data and files 

The required files for the analysis presented in the paper are provided in this repository. 

The analysis of water security by intra-city sectors requires a _GeoDataFrame_ of scores (normalised between 0 and 1 in this project, according to thresholds as described in the supplementary information provided with the paper). The file **SectorsScores.shp** should contain columns indicating a unique ID or code for the sector,  the geometry of the sector's polygon, the population of each sector and the indicators. The code for the indicators must contain the letter and numbers following the hierarchical organization of the framework (for example: a12, a21, b11,c22,d12, etc).   A file with the results for the entire city is also required (**CityScores.xls**). For this file, all that is needed is a score attributed to the entire city for each of the indicators.

The **Sample_size.xls** is a file that provides the sample size for each indicator and description of the indicator (name, dimension, category, id, etc). 
The number of divisions of the intra-city boundary for which data is found is  considered as the sample size ($n$) for that measure. For example, an indicator where only one measure was available for the entire city boundary had a sample size of 1. 

A **Base Map** is also required for the map plots. 

## License 

Creative Commons Zero v1.0 Universal

## Contact
Juliana Marcal (jm2842@bath.ac.uk)
Jan Hofman (j.a.h.hofman@bath.ac.uk)

> **Note:** This work was conducted as part of the Water Informatics Science and Engineering (WISE) Centre for Doctoral Training (CDT), funded by the UK Engineering and Physical Sciences Research Council, Grant No.EP/L016214/1. Juliana Marcal is supported by a research studentship from this CDT. 
