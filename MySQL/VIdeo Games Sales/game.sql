CREATE DATABASE video_game;
USE video_game;
CREATE TABLE test1(
test TEXT
);

SELECT * FROM vgsales_original;

SELECT DISTINCT(Platform) FROM vgsales_original;
SELECT DISTINCT(Genre) FROM vgsales_original;
SELECT DISTINCT(Publisher) FROM vgsales_original;
SELECT NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales FROM vgsales_original;

ALTER TABLE game
ADD PRIMARY KEY(game_ID);

ALTER TABLE sales
ADD FOREIGN KEY (Game_ID) REFERENCES game(game_ID);

SELECT * FROM game gm, sales s, platform p, genre g, publisher pb
WHERE gm.game_ID = s.Game_ID AND gm.platform_ID = p.platform_ID AND gm.genre_ID = g.genre_ID AND gm.publisher_ID = pb.publisher_ID;

SELECT * FROM game gm
INNER JOIN platform p ON gm.platform_ID = p.platform_ID
INNER JOIN genre g ON gm.genre_ID = g.genre_ID
INNER JOIN publisher pb ON gm.publisher_ID = pb.publisher_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID;

SELECT gm.Name, s.NA_Sales AS North_America, s.EU_Sales AS Europe, s.JP_Sales AS Japan, s.Other_Sales AS others, s.Global_Sales AS Global
FROM game gm
INNER JOIN sales s ON gm.game_ID = s.Game_ID;

SELECT platform, ROUND(SUM(s.NA_Sales),2) AS North_America, ROUND(SUM(s.EU_Sales),2) AS Europe, ROUND(SUM(s.JP_Sales),2) AS Japan, ROUND(SUM(s.Other_Sales),2) AS others, ROUND(SUM(s.Global_Sales),2) AS Global FROM platform p
INNER JOIN game gm ON p.platform_ID = gm.platform_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID
GROUP BY platform
ORDER BY global DESC;

SELECT platform FROM platform
WHERE platform LIKE 'X%' OR platform LIKE 'PS%';

#i)
SELECT gm.name, gm.Year, p.platform FROM game gm
INNER JOIN platform p ON gm.platform_ID = p.platform_ID
WHERE Year = 2001 AND platform = 'GBA';

#ii)-NA
SELECT p.platform, ROUND(SUM(NA_Sales),2) AS North_America FROM platform p
INNER JOIN game gm ON p.platform_ID = gm.platform_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID
GROUP BY p.platform
ORDER BY North_America DESC;

#ii)-EUROPE
SELECT p.platform, ROUND(SUM(EU_Sales),2) AS Europe FROM platform p
INNER JOIN game gm ON p.platform_ID = gm.platform_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID
GROUP BY p.platform
ORDER BY Europe DESC;

#iii)
SELECT platform, ROUND(SUM(s.NA_Sales),2) AS North_America, ROUND(SUM(s.EU_Sales),2) AS Europe, ROUND(SUM(s.JP_Sales),2) AS Japan, ROUND(SUM(s.Other_Sales),2) AS others, ROUND(SUM(s.Global_Sales),2) AS Global FROM platform p
INNER JOIN game gm ON p.platform_ID = gm.platform_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID
WHERE platform LIKE 'X%' OR platform LIKE 'PS%'
GROUP BY platform
ORDER BY global DESC;

#iv)
SELECT g.genre, ROUND(SUM(Other_Sales),2) AS Others_region FROM genre g
INNER JOIN game gm ON g.genre_ID = gm.platform_ID
INNER JOIN sales s ON gm.game_ID = s.Game_ID
GROUP BY genre
ORDER BY Others_region DESC; 

SELECT Name,ROUND(SUM(NA_Sales+JP_Sales+Other_Sales+Global_Sales),2) Total_Sales
FROM game
JOIN sales ON sales.Game_ID=game.game_ID
WHERE year<2004
GROUP BY Name
ORDER BY Total_Sales DESC;

SELECT g.name, ROUND(SUM(global_sales),2) AS total_sales
FROM game AS g JOIN sales AS s
ON g.game_ID = s.game_ID
GROUP BY name
ORDER BY total_sales DESC
LIMIT 10;

SELECT g.name, ROUND(SUM(global_sales),2) AS total_sales
FROM game AS g JOIN sales AS s
ON g.game_ID = s.game_ID
GROUP BY name
ORDER BY total_sales DESC
LIMIT 10;

SELECT * FROM game;

SELECT gr.genre, ROUND(STD(NA_Sales),2) AS North_America, ROUND(STD(EU_Sales),2) AS Europe, ROUND(STD(JP_Sales),2) AS Japan, ROUND(STD(Other_Sales),2) AS Others, ROUND(STD(Global_Sales),2) AS Global
FROM genre gr 
JOIN game g ON gr.genre_ID = g.genre_ID
JOIN sales s ON g.game_ID = s.game_ID
GROUP BY genre
ORDER BY genre;

CREATE TABLE sales (
	FOREIGN KEY(game_ID) REFERENCES game(game_ID),
    NA_Sales double,
    EU_Sales double,
    JP_Sales double,
    Other_Sales double,
    Global_Sales double
    );