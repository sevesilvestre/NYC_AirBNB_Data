# New York AirBNB Data Science Project: Project Overview
• Cleaned and managed a complex dataset of 50,000 data points and 16 variables to gather relevant and insightful data in Python

• Developed nine different models utiliizing cross validation, dimensionality reduction, and K-Modes/K-Prototype clustering to determine which variables affected NYC AirBNB prices while providing business insight on the NYC AirBNB environment

• Concluded that NYC AirBNB’s are influenced most by location and room type where “Entire Home” AirBNB’s in Manhattan increased the price of an AirBNB the most

## Code Used:
Python Version: 3.8

Packages: pandas, numpy, sklearn, matplotlib, kmodes

## Challenges: 
None of the clustering models learned throughout my class was working with the data I was working with. (K-Means, Gaussian, DBSCAN, and Hierarchical Clustering). I refused to give up on the clustering model, since I believed that it would produce great value if performed, and I had to teach myself on my own using external resources of a new clustering method that I wasn't sure was going to work, which is where I learned of K-Prototype and implemented it in my work.


## Results:
Concluded that AirBNB listings that provide the customer an entire home/apartment tended to cause prices to jump, while shared rooms tended to cause prices to dip. Additionally, we were able to determine how the borough the AirBNB is located can affect the price. Obviously, Manhattan would generate a higher price, but an AirBNB in either Staten Island or Bronx both negatively affected the price at the same magnitude while one in Queens negatively affected the price less and one in Brooklyn positively affected the total price of the Airbnb listing.

The K-Prototype clustering method determined 3 clusters of AirBNB customers: 
- Cluster 1: Customers who don’t care what type of room they book, as long as it is the cheapest price. 
- Cluster 2: Customers who mainly book Entire homes and apartments potentially to fit larger parties of guests, but at not too high of a cost. 
- Cluster 3: Very wealthy customers who have big money and spend approximately $6,000 a night on an AirBNB.

