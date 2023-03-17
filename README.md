<h1 align="center">New York AirBNB Data Science Project </h1>

<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/nyc.webp">
</p>

## Overview
With New York having the 3rd most AirBNB listings in 2021 with over 94,000 listings, this project delves into the factors that influence New York City's AirBNB prices, using advanced modeling techniques such as cross-validation, dimensionality reduction, and K-Modes/K-Prototype clustering. With a dataset containing 50,000 data points and 16 variables, we used Python Notebook to clean and visualize the data, and identified key drivers of price fluctuations in one of the world's busiest markets for short-term rentals.

## Code Used:
Python Version: 3.8

Packages: pandas, numpy, sklearn, matplotlib, kmodes

## The questions that I aimed to address through my data project are:
- How are AirBNB rentals distributed among the five boroughs of New York?
- What is the price distribution of AirBNB rentals among the five boroughs?
- What can we learn from a clustering model of all brooklyn listings using "price" and "room type" as variables?

## Key Findings
- Concluded that AirBNB listings that provide the customer an entire home/apartment tended to cause prices to jump, while shared rooms tended to cause prices to dip. 

- Manhattan would generate a higher price, but an AirBNB in either Staten Island or Bronx both negatively affected the price at the same magnitude while one in Queens negatively affected the price less and one in Brooklyn positively affected the total price of the Airbnb listing.

- 3 clusters of AirBNB customers:
  - Cluster 1: Customers who don’t care what type of room they book, as long as it is the cheapest price.
  - Cluster 2: Customers who mainly book Entire homes and apartments potentially to fit larger parties of     guests, but at not too high of a cost.
  - Cluster 3: Very wealthy customers who have big money and spend approximately $6,000 a night on an AirBNB.

## Data Gathering
With these questions I had, I downloaded a dataset from Kaggle to help me answer these questions:
- **[New York City AirBnB Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)** (Source: Kaggle)

## Data Cleaning + Data Manipulation
After loading the dataset into Jupyter Notebooks, the first order of business is drop any columns that will not be necessary fro this analysis, as well as creating dummy variables for each borough as the "borough" variable will be used later on in regression and cluster analysis.

```
nyc = nyc.drop(["latitude", "longitude", "last_review", "host_name", "id", "host_id", "name"], axis = 1)
```
```
dummies_neighbourhood = pd.get_dummies(nyc["neighbourhood_group"])
dummies_room = pd.get_dummies(nyc["room_type"])

nyc = pd.concat([nyc,dummies_neighbourhood], axis = 1)
nyc = pd.concat([nyc,dummies_room], axis = 1)

nyc = nyc.dropna()
```
## Question 1: How are AirBNB rentals distributed among the five boroughs of New York?
Using ggplot in Python, we are able to create a bar chart of the number of AirBNB's present in each borough as well as a pie chart displaying the proportion of listings in each borough using the following code:
```
ggplot(nyc, aes(x = "neighbourhood_group", fill = "neighbourhood_group")) + geom_bar() + theme_minimal() + theme(panel_grid_major = element_blank()) + labs(title = "How Many listings are there in each borough?")
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/nycbar.png">
</p>

We can see that out the 5 boroughs, Manhattan has the most number of listings and Brooklyn has the 2nd most highest number of listings. On the other hand, Staten Island has the least number of AirBNB listings and Bronx comes in with the 2nd lowest number of listings.

```
listings = nyc['neighbourhood_group'].value_counts()
plot = listings.plot.pie(y='count', figsize=(5, 5), autopct='%1.0f%%')
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/mycpie.png">
</p>

We can see that Manhattan contains 43% of all NYC AirBNB listings while Staten Island only contains 1% of all the listings. We can see that the majority of this dataset is consumed of listings located in Manhattan and Brooklyn making up 85% of all data in this dataset.

## Question 2: What is the price distribution of AirBNB rentals among the five boroughs?
To analyze the price distribution of AirBNB rentals among the five boroughs, a linear regression model was set up to see the relationship between continous variables in our data set and price. The following code displays the process of setting up a linear regression model as well as plot to analyze the correlation coefficients per variable against price: 
```
predictors = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365","Bronx", "Brooklyn","Manhattan","Queens","Staten Island", "Entire home/apt","Private room", "Shared room"]
#cont = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]
X = nyc[predictors]
y = nyc["price"]

z = StandardScaler()
X[["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]] = z.fit_transform(X[["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]])

lr = LinearRegression()
lr.fit(X,y)

coefficients = pd.DataFrame({"Coef":lr.coef_,
              "Name": predictors})
```
```
ggplot(coefficients, aes(x = "Name" , y = "Coef", fill = "Name")) + geom_bar(stat = "identity") + theme_minimal() + labs(title = "NYC Predictor Coefficient Values") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7, angle = 90))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/NYCCoefficient.png">
</p>

This bar chart displays the coefficient values of each value on their affect on AirBNB prices. The higher the coefficient, the stronger the positive relationship there is between these variables. The more negrative the coefficient, the stronger the negative relationship there is between these variables. Based on this chart, we notice that if a listing is an Entire home/apt, that tends to bump up the price the most compared to any other variable. Another variable that will increase price is if the listing is located in Manhattan. Variables such as Shared room and Staten Island present a negative coefficient value meaning that if a listing contains these variables, they tend to decrease the price of a listing. Variables that are valued at 0 tend to have 0 effect on the price of an AirBNB listing such as availability and number of reviews.

As this bar chart represents the affect of variables on price in NYC as a whole, an more in-depth analysis was performed where a linear regression model was performed per borough located  [here](https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/NYC_AirBNB.ipynb).

After determining which varaibales affect price the most and least, a bar chart of average prices of AirBNB's in each borough was developed using the following code:
```
brooklynprice = Brooklyn["price"].mean()
bronxprice = Bronx["price"].mean()
manhattanprice = Manhattan["price"].mean()
queensprice = Queens["price"].mean()
statenprice = Staten["price"].mean()

averageprice = {"Brooklyn": brooklynprice, "Bronx": bronxprice, "Manhattan": manhattanprice, "Queens": queensprice,
               "Staten Island": statenprice}

plt.bar(list(averageprice.keys()), averageprice.values(), color = "orange")
plt.xlabel('Borough')
plt.ylabel('Price')
plt.title('Average Price of an AirBNB in each Borough')
plt.show()

print("Average price in Brooklyn:", brooklynprice)
print("Average price in Bronx:", bronxprice)
print("Average price in Manhattan:", manhattanprice)
print("Average price in Queens:", queensprice)
print("Average price in Staten Island:", statenprice)
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/nycprice.png">
</p>

The price distribution of AirBNB rentals across the five buroughs are 180/night in Manhattan, 121/night in Brooklyn, 96/night in Queens, 90/night in Staten Island, and 80/night in the Bronx.

## Question 3: What can we learn from a clustering model of all brooklyn listings using "price" and "room type" as variables?

One challenge I faced throughout this project was that none of the clustering models learned throughout my class (K-Means, Gaussian, DBSCAN, and Hierarchical Clustering) was working with the data I was working with since I was trying to create a clustering model combining both categorical and continous data. I refused to give up on the clustering model, since I believed that it would produce great value if performed, and I had to teach myself on my own using external resources of a new clustering method that I wasn't sure was going to work, which is where I learned of K-Prototype and implemented it in my work.

The following code shows the process of implementing the K-Prototype clustering model into my data as well as the graphs that were produced to display each cluster:
```
kproto = KPrototypes(n_clusters = 3, verbose = 2, max_iter = 20)
clusters = kproto.fit_predict(brooklyn_array, categorical = [0])
```
```
cluster_dict = []
for c in clusters:
    cluster_dict.append(str(c))
    
Brooklyndrop['cluster'] = cluster_dict
```
### Cluster 0: Lowest Price Cluster
```
ggplot(cluster0, aes(x = "room_type", y = "price")) + geom_boxplot() + theme_minimal() + labs(title = "Cluster 0 of Brooklyn AirBNB users") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/NYCCluster.png">
</p>

This boxplot displays the price distribution of each room type for our first cluster, "Cluster 0". Customers in this cluster tend to spend around 131/night for an entire home/apartment, around 68/night for a private room, and around 43/night for a shared room. The dots on on each boxplot also display any outliers on certain data points that fit in our cluster but are not near the mean price for each room type.

```
ggplot(cluster0, aes(x = "room_type", fill = "room_type")) + geom_bar() + theme_minimal() + labs(title = "Number of Listings in Cluster 0 for each Room Type") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/cluster0coef.png">
</p>

This bar chart describes the number of listings for each room type in our first cluster, "Cluster 0". Customers who rented out private rooms consisted of most of the data points in this cluster while Entire homes and apartments were rented out second most in this cluster. Not many customers in this cluster rented out a shared room comapred to the other room types, but this cluster represents most of all renters who rented our shared rooms in this whole dataset as we will see in the next charts.

### Cluster 1: Entire Home/Apt Only Cluster
```
ggplot(cluster1, aes(x = "room_type", y = "price")) + geom_boxplot() + theme_minimal() + labs(title = "Cluster 1 of Brooklyn AirBNB users") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/cluster1.png">
</p>

This boxplot displays the price distribution of each room type for our second cluster, "Cluster 1". Customers in this cluster tend to spend around 346/night for an entire home/apartment, around 445/night for a private room, and around 292/night for a shared room. The dots on on each boxplot also display any outliers on certain data points that fit in our cluster but are not near the mean price for each room type. We see that prices for Entire home and apartment in this cluster can go all the way to 2500/night.

```
ggplot(cluster1, aes(x = "room_type", fill = "room_type")) + geom_bar() + theme_minimal() + labs(title = "Number of Listings in Cluster 1 for each Room Type") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/cluster1coef.png">
</p>

This bar chart describes the number of listings for each room type in our second cluster, "Cluster 1". Customers who rented out Entire homes and apartments consisted of most of the data points in this cluster while both private rooms and shared rooms were barely rented out in this cluster.

### Cluster 2: High Spenders
```
ggplot(cluster2, aes(x = "room_type", y = "price")) + geom_boxplot() + theme_minimal() + labs(title = "Cluster 2 of Brooklyn AirBNB users") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/cluster2.png">
</p>

This boxplot displays the price distribution of each room type for our third cluster, "Cluster 2". Customers in this cluster tend to spend around 6500/night for an entire home/apartment, around 6250/night for a private room, and no customers in this cluster rented out a shared room. The box for an entire home/apartment shows that customers in this cluster are willing to pay from 5000 to 8000 per night and 5750 to 7000 per night for a private room.

```
ggplot(cluster2, aes(x = "room_type", fill = "room_type")) + geom_bar() + theme_minimal() + labs(title = "Number of Listings in Cluster 2 for each Room Type") + theme(panel_grid_major_x = element_blank(), 
      panel_grid_major_y = element_blank(), 
      panel_grid_minor_y = element_blank(), 
       axis_text_x = element_text(size = 7))
```
<p align="center">
  <img src= "https://github.com/sevesilvestre/NYC_AirBNB_Data/blob/main/images/cluster2coef.png">
</p>

This bar chart describes the number of listings for each room type in our third cluster, "Cluster 2". This cluster only contains 5 customers who rented out an entire home/apartment and only 2 customers who rented out a private room. No customer in this cluster rented a shared room.


## Results:
After analyzing this data and answering each question, we concluded that AirBNB listings that provide the customer an entire home/apartment tended to cause prices to jump, while shared rooms tended to cause prices to dip. Additionally, we were able to determine how the borough the AirBNB is located can affect the price. Obviously, Manhattan would generate a higher price, but an AirBNB in either Staten Island or Bronx both negatively affected the price at the same magnitude while one in Queens negatively affected the price less and one in Brooklyn positively affected the total price of the Airbnb listing.

The K-Prototype clustering method determined 3 clusters of AirBNB customers: 
- Cluster 1: Customers who don’t care what type of room they book, as long as it is the cheapest price. 
- Cluster 2: Customers who mainly book Entire homes and apartments potentially to fit larger parties of guests, but at not too high of a cost. 
- Cluster 3: Very wealthy customers who have big money and spend approximately $6,000 a night on an AirBNB.

Overall, these insights would be of great value for AirBNB hosts looking to gain market knowledge of the NYC ecosystem and correctly position themselves for the most return possible.
