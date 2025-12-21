import pandas as pd
import matplotlib.pyplot as plt

data = [1, 2, 4, 6, 8, 10, 12, 11, 9, 7, 5, 3] 
df = pd.DataFrame({'data': data})

# Calculate rolling standard deviation
roll_std = df['data'].rolling(5).std()

# Threshold for motion detection 
threshold = 1

# Mark motion if rolling std exceeds threshold
df['motion'] = (roll_std > threshold).astype(int) 

# Plot original data
plt.plot(df['data'], label='Original Data')

# Plot motion markers
plt.plot(df.index, df['motion']*max(df['data']), 'ro', label='Motion Detected')

plt.xlabel('Time')
plt.ylabel('Value') 
plt.title('Motion Detection using Rolling Std')
plt.legend()
plt.show()