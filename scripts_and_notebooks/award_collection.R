library(baseballr) # for collecting the awards winners
library(tidyverse) # for data manipulation

## Rookie of the Year Data using `baseballR`

awards <- mlb_awards()
# various award names
award_names <- unique(awards$award_name)

# Extract the award names that contain the word "Rookie"
rookie_awards <- award_names[str_detect(award_names, "Rookie")]

# Get the award ids for the Rookie awards
rookie_id <-
  awards %>% 
    filter(award_name %in% c("Jackie Robinson AL Rookie of the Year", 
                             "Jackie Robinson NL Rookie of the Year")) %>% 
    select(award_id)

# Split the awards into American League and National League

## American League (AL)
roy_AL <- mlb_awards_recipient(award_id = "ALROY")
roy_AL <- roy_AL %>% select(award_id, season, player_id)

## National League (NL)
roy_NL <- mlb_awards_recipient(award_id = "NLROY")
roy_NL <- roy_NL %>% select(award_id, season, player_id)

# Combine the results into one larger dataframe
roy <- rbind(roy_AL, roy_NL)

# Create a csv for the award data
write_csv(roy, "roy_awards.csv")

### Rookie of the Year Voting using Lahman Data
### Lahman Database includes detailed voting measures including:
### - Points Won
### - Points Max
### - vote_share
### - unanimous

lahman <- read_csv("../Lahman Database (04-01-24)/AwardsSharePlayers.csv")

df <- 
  lahman %>% 
  filter(yearID %in% c(1949:2023), # Rookie of the Year awards began AL/NL in 1949
         awardID == "Rookie of the Year") %>% # Filter for Rookie of the Year voting only
  arrange(yearID, desc(pointsWon)) %>% 
  mutate(unanimous = ifelse(pointsWon == pointsMax, 1, 0)) # Create a column for unanimous winners

df <- 
  df %>%
  group_by(yearID, lgID) %>% 
  # Create a column for the Rookie of the Year winner
  mutate(rookie_of_the_year = if_else(pointsWon == max(pointsWon), 1, 0)) %>% 
  ungroup() %>% 
  arrange(desc(rookie_of_the_year), yearID)
df 

# Create a csv with the detailed Rookie of the Year Voting
write_csv(df, "data/roy_voting.csv")