import pandas as pd

data = [
    ("bet365.com", 1), ("williamhill.com", 1), ("partycasino.com", 1), ("stake.com", 1), ("mrgreen.com", 1),
    ("theguardian.com", 0), ("wikipedia.org", 0), ("amazon.com", 0), ("twitch.tv", 0), ("techcrunch.com", 0),
    ("ro.williamhill.com", 1), ("superbet.ro", 1), ("fortuna.ro", 1), ("maxbet.ro", 1), ("unibet.ro", 1),
    ("digi24.ro", 0), ("hotnews.ro", 0), ("emag.ro", 0), ("olx.ro", 0), ("libertatea.ro", 0),
    ("888casino.com", 1), ("pokerstars.com", 1), ("10bet.com", 1), ("ladbrokes.com", 1), ("casumo.com", 1),
    ("nytimes.com", 0), ("cnn.com", 0), ("linkedin.com", 0), ("github.com", 0), ("bbc.com", 0),
    ("netbet.ro", 1), ("winmasters.ro", 1), ("betano.ro", 1), ("getsbet.ro", 1), ("gameworld.ro", 1),
    ("stirileprotv.ro", 0), ("ziaruldeiasi.ro", 0), ("adevarul.ro", 0), ("evz.ro", 0), ("antena3.ro", 0),
    ("draftkings.com", 1), ("fanduel.com", 1), ("casino.org", 1), ("slots.lv", 1), ("vegasplus.com", 1),
    ("stackoverflow.com", 0), ("duckduckgo.com", 0), ("imdb.com", 0), ("ycombinator.com", 0), ("wired.com", 0)
]

df = pd.DataFrame(data, columns=["url", "label"])
df.to_csv("html_dataset.csv", index=False)
