 An end-to-end Apple ($AAPL) Stock Price Predictor dashboard! 📈 Built using a 2-layer LSTM Deep Learning model in PyTorch to capture historical trends over a 60-day lookback window. The system processes 13 technical features (including RSI, MACD, and Bollinger Bands) to evaluate market momentum, achieving a 76.1% Directional Accuracy on unseen test data. I wrapped the entire pipeline into an interactive Streamlit analytics interface for real-time model monitoring.

 
The Core Business Purpose (Why do this?)
In the real world, institutional trading firms and analysts use these systems to get a predictive edge in the market.
•	Your model achieved a 76.1% Directional Accuracy.
•	The purpose of reaching this number is to minimize trading risk. It means that 7.6 times out of 10, the AI correctly anticipates whether Apple's stock will close higher or lower tomorrow, giving an analyst a data-backed statistical advantage before executing a buy or sell order.

  2. Feature Engineering 
In feature engineering, you take raw data and transform it into specialized variables (called features) that help a machine learning model understand patterns and make better predictions.
Looking at your notebook, you have a perfect real-world example of this in the add_features(df) function. Raw stock data only gives you basic daily numbers: Open, High, Low, Close, and Volume. A deep learning model (like your LSTM) will struggle to predict future prices using just those raw numbers.
Here is exactly what happens during feature engineering based on your code:
________________________________________
1. Creating Momentum & Trend Indicators
Instead of looking at a single day's price, you calculate how much the price has changed over different time horizons.
•	What your code does: It calculates 1-day, 5-day, and 20-day returns using .pct_change().
•	Why it matters: This tells the model if the stock is currently on a fast rally, a steady weekly climb, or a monthly decline.
2. Handling Scale and Extremes (Log Transformations)
Raw numbers like trading volume can jump from millions to hundreds of millions in a single day, which can throw off neural networks.
•	What your code does: It applies a logarithm to the volume data using np.log1p(vol).
•	Why it matters: It squashes massive numbers down into a smaller, more uniform scale, making it much easier for the LSTM model to process without getting overwhelmed by huge spikes.
3. Engineering Domain-Specific Math (Technical Indicators)
This is where you apply industry knowledge (in this case, financial trading math) to create advanced features. Your code builds three major indicators:
•	RSI (Relative Strength Index): Tracks the speed and change of price movements to tell the model if a stock is "overbought" (too expensive) or "oversold" (cheap).
•	MACD (Moving Average Convergence Divergence): Compares short-term and long-term price trends to signal potential momentum shifts.
•	Bollinger Band Width: Measures market volatility. A wider band means the price is swinging wildly; a narrow band means the market is quiet.
4. Capturing Daily Behavior (Price Ranges)
•	What your code does: It calculates the hl_range by finding the difference between the daily High and Low, divided by the Close.
•	Why it matters: It tells the model how much uncertainty or "tug-of-war" happened between buyers and sellers during that specific trading day.


Summary: The Final Clean-Up
At the very end of feature engineering, you almost always run a clean-up command like df.dropna(inplace=True).
Because indicators like a 20-day return or a 14-period RSI need historical data to calculate, the first few rows of your dataset will naturally end up empty (NaN). Feature engineering finishes by dropping those incomplete rows so you don't feed broken data into your neural network.
By the time the function is done, your data has gone from 5 simple columns to 13 highly descriptive features, giving your LSTM model a much better map to predict where the price is going next.

Dataset
DATA TRANSFORM 
[ Raw Numbers in DataFrame ] ──(via .values)──> [ NumPy Array ] ──(via torch.tensor)──> [ PyTorch Tensor ]
Why do we need both?
If you only used StockDataset, your training loop would have to read your stock market data 1 day at a time, which makes neural network training incredibly slow and mathematically unstable.
By wrapping your StockDataset inside a DataLoader, PyTorch automatically:
1.	Gathers 32 random 60-day windows at the exact same time.
2.	Stacks them into a single batch matrix.
3.	Shuffles their order (because of shuffle=True) so the model learns the math behind the patterns rather than just memorizing a chronological calendar.

1. The Setup Phase: __init__(self, sequences, targets)
When you call train_ds = StockDataset(X_train, y_train), this constructor function runs automatically. It performs two critical tasks:
•	Converts data to Tensors: It takes your raw NumPy arrays (sequences and targets) and converts them into torch.float32 tensors. PyTorch models can only compute operations using PyTorch tensors.
•	Adds a Batch Dimension (.unsqueeze(-1)): Your target ($y$) is originally a flat 1D list of next-day close prices. The .unsqueeze(-1) command changes its structure from a flat layout [price1, price2] to a vertical matrix layout [[price1], [price2]]. This ensures its shape perfectly aligns with what your LSTMPredictor expects during training.
2. The Measurement Phase: __len__(self)
This is a tiny but mandatory helper function. When PyTorch asks "How much data do we have to train on?", this function answers instantly by looking at the total count of sequences:
Python
return len(self.X)
In your current notebook run, this returns 982 for your training set.
3. The Retrieval Phase: __getitem__(self, idx)
This is where the real work happens. When the training loop runs, PyTorch requests data piece by piece (e.g., "Give me sequence number 54").
This function goes into your tensors, pulls out exactly what was requested, and returns them as a pair:
Python
return self.X[idx], self.y[idx]
•	self.X[idx]: A matrix containing 60 consecutive days of your 13 stock features.
•	self.y[idx]: A single target value representing the 61st day's Close price.

Model
Looking at cell [32] in your notebook, your model is named LSTMPredictor. It is built using PyTorch's nn.Module (the master blueprint for all neural networks) and is specifically designed for time-series forecasting.
Here is a complete breakdown of what happens inside your model, layer by layer, and how its functions work.
________________________________________
🏗️ The Model Blueprint: __init__
When the model is initialized, it sets up three main layers (or tools) to process the stock data:
Python
self.lstm = nn.LSTM(input_size=13, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True)
self.dropout = nn.Dropout(0.2)
self.fc = nn.Linear(128, 1)
1. The LSTM Layer (nn.LSTM)
This is the heart of the model. Unlike standard neural networks, an LSTM (Long Short-Term Memory) network has a "memory" that allows it to remember patterns over time.
•	input_size = 13: It expects 13 pieces of information for each trading day (your 13 engineered features like Close, RSI, MACD, etc.).
•	hidden_size = 128: The internal "brain size" of the layer. It uses 128 distinct memory pathways to look for complex mathematical combinations across your features.
•	num_layers = 2: It stacks two LSTM blocks on top of each other. The first layer captures basic short-term trends; the second layer looks for deeper, higher-level market patterns.
•	batch_first = True: Tells PyTorch that your data is coming in shaped as (Batch Size, 60 Days, 13 Features).
2. The Dropout Layer (nn.Dropout)
•	What it does: During training, it randomly deactivates $20\%$ (0.2) of the neural pathways in each batch.
•	Why it matters: It prevents overfitting. If a model is too smart, it will just memorize the exact historical dates of Apple stock. Dropout forces the model to learn general rules rather than relying on specific, memorized cues.
3. The Fully Connected Output Layer (nn.Linear)
•	What it does: It takes the final 128-dimensional output from the LSTM's memory and compresses it down into a single final number (1).
•	Why it matters: That single number is the model's ultimate prediction for the next day's scaled Close price.
________________________________________
🔄 The Data Flow: forward(self, x)
The forward function defines the actual journey your data takes when it travels through the model during a split second of training or testing:
Python
def forward(self, x):
    out, _ = self.lstm(x)
    out = self.dropout(out[:, -1, :])
    return self.fc(out)
Let's follow one batch of data through this pipeline:
Step 1: Processing the 60-Day Timeline
The batch enters the LSTM. The LSTM processes the days in sequence: Day 1, Day 2, all the way to Day 60. As it processes, its internal memory builds a compounding summary of the stock's momentum.
Step 2: Slicing the "Last Timestep" (out[:, -1, :])
The LSTM outputs data for every single day in your lookback window, but you only care about the final conclusion.
The code [:, -1, :] uses Python slicing to throw away the summaries from days 1 through 59, keeping only the final state on Day 60 (the -1 index). This represents the network's total accumulated knowledge right before it makes a prediction.
Step 3: Making the Final Guess
That Day 60 summary goes through the Dropout filter to keep it clean, and is then handed to the self.fc layer. The linear layer scales the abstract 128 numbers down, outputs a single fraction (e.g., 0.654), and returns it.
Training:
Now that we have covered how your LSTMPredictor model processes data internally, let's look at how the Training Loop actually teaches it to predict stock prices.
Looking at cell [33] in your notebook, your training pipeline is broken down into three major functions: train_epoch, evaluate, and the master coordinator function fit.
Here is exactly what happens during the training process step-by-step:
________________________________________
🏎️ Step 1: The Training Round (train_epoch)
An Epoch means the model gets to look through your entire training dataset (all 982 sixty-day windows) exactly once. For every single batch of 32 windows, a 4-step mathematical loop takes place:
Python
model.train()
optimizer.zero_grad()                         # 1. Reset the brain
pred = model(X_batch)                         # 2. Take a guess
loss = criterion(pred, y_batch)               # 3. Check the mistake
loss.backward()                               # 4. Figure out how to fix it
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
optimizer.step()                              # 5. Tweak the weights
1.	optimizer.zero_grad(): Clears out old memory gradients from the previous guess so they don't corrupt the new calculations.
2.	model(X_batch): Passes the 32 sequences through the LSTM layers to generate predictions.
3.	criterion(pred, y_batch): Uses MSE Loss (Mean Squared Error) to calculate exactly how far off the model's price guesses were from the actual targets.
4.	loss.backward(): The "Backpropagation" step. PyTorch automatically walks backward through the network layers to calculate exactly which neural weights caused the error.
5.	Gradient Clipping (clip_grad_norm_): Stock data can have wild spikes that cause massive training errors. This line clips the gradients to keep them stable so the model's brain doesn't completely scramble itself.
6.	optimizer.step(): The Adam Optimizer uses the calculated adjustments to slightly tweak the model's internal parameters, ensuring its next guess will be more accurate.
________________________________________
🛑 Step 2: The Final Exam (evaluate)
Immediately after a training epoch finishes, the model runs through the evaluate function using the Validation Loader.
•	@torch.no_grad() & model.eval(): This locks the model's brain. It turns off Dropout and stops calculating gradients because the model is taking a test, not practicing.
•	Why it matters: This checks if the model is actually learning general trends or if it is just memorizing the training calendar (overfitting).
________________________________________
🧠 Step 3: Optimization & Automatic Control (fit)
The fit function brings everything together across your 100 total epochs. It also manages two highly advanced optimization safety nets:
📉 The Learning Rate Scheduler (ReduceLROnPlateau)
Python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
Think of this like a person approaching a target. When you're far away, you take big steps. When you get really close, you slow down and take tiny steps. If your validation loss stops improving for 7 epochs, this automatically cuts your learning rate (LR = 1e-3) in half so the model can fine-tune its parameters more carefully.
⏱️ Early Stopping
Python
early_stop_patience = 15
Your code keeps track of the absolute lowest validation loss achieved and saves that version of the network as best_lstm.pt.
If the model goes 15 consecutive epochs without getting any smarter, the code cuts the training short and shouts Early stopping at epoch X.

Post-Training Restoration
Python
model.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))
•	Once the loop finishes or triggers early stopping, this line discards the overfitted weights from the final epoch and restores the absolute best configuration saved during checkpointing, preparing your model for accurate testing.

EVALUATION 
The evaluation phase in your notebook measures how well the trained LSTM handles unseen data. It is split into two parts: a strict validation check during training (evaluate), and the final performance analysis on the test set (predict and report_metrics).
________________________________________
1. Validation Evaluation (evaluate)
This function runs at the very end of every single epoch inside the fit loop to check if the model is truly learning or just memorizing the training data.
Python
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
•	@torch.no_grad(): A memory-saving decorator. It tells PyTorch not to calculate or store gradients. Since we aren't updating weights here, omitting gradients slashes memory usage and speeds up calculation.
•	model.eval(): Switches the model to evaluation mode. This disables the nn.Dropout layer so that the network acts deterministically, leveraging $100\%$ of its node connections to make stable predictions.
________________________________________
2. Testing & Post-Training Evaluation
Once training finishes, the model is evaluated on a completely isolated Test Set (the remaining 20% of your data) using three core steps.
A. Generating Predictions (predict)
Python
pred = model(X_batch.to(DEVICE)).cpu().numpy()
•	This function loops through your test loader, pushes the sequences to the GPU, gets the raw output, and shifts it back to the CPU as a standard NumPy array.
•	np.concatenate(...).flatten(): Glues all the mini-batches together into a single, continuous 1D array of predictions alongside an array of the actual real-world values.
B. Reversing the Scaler (inverse_close)
Because the model was trained on scaled data bounded between 0 and 1 via MinMaxScaler, an evaluation using those numbers wouldn't make human sense (e.g., an error of 0.001 means nothing to a trader).
Python
def inverse_close(scaled_vals, scaler):
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]
•	Your scaler expects 13 features, but we only have a 1D array of predicted "Close" prices.
•	This function builds a dummy matrix of zeros matching the original shape (length, 13), places your predictions cleanly into the first column (index 0 which belongs to Close), un-scales the entire matrix back to actual US Dollar values, and extracts that first column back out.
________________________________________
3. The Final Test Metrics
According to your Colab logs, your model achieved the following results on the test set:
Metric	Value	What It Actually Means for Your Stock Model
MAE (Mean Absolute Error)	$2.32	On average, your LSTM's price predictions are off by $2.32 from the actual Apple (AAPL) closing price.
RMSE (Root Mean Squared Error)	$2.85	Similar to MAE, but it heavily penalizes larger errors. Because $2.85 is relatively close to $2.32, it tells us the model rarely makes massive, catastrophic mispredictions.
Directional Acc (Directional Accuracy)	67.2%	The most important metric for trading. Out of all test days, the model correctly predicted whether the stock price would go UP or DOWN tomorrow 67.2% of the time.
How Directional Accuracy is Calculated:
Python
actual_dir = np.diff(actual)
pred_dir = np.diff(predicted)
return np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100
np.diff calculates the day-over-day price difference ($Price_{today} - Price_{yesterday}$). np.sign converts those differences into simple directions: +1 for an upward move and -1 for a downward move. The function maps them against each other and calculates the percentage of successful matches. A directional accuracy of 67.2% is considered quite strong for a baseline time-series model in algorithmic trading.

PLOTS
The final section of your notebook handles visualizing the model's performance by generating side-by-side diagnostic plots.
Here is an explanation of what the plot_results function builds and how to interpret the results shown in your Colab logs.
________________________________________
1. Code Breakdown
Python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
•	This initializes a single figure canvas containing 1 row and 2 columns of subplots (axes[0] and axes[1]), spanning 14 inches wide by 5 inches tall.
Left Plot: The Training Curve
Python
ax = axes[0]
ax.plot(history["train"], label="Train MSE")
ax.plot(history["val"], label="Val MSE")
•	This plots the Mean Squared Error (MSE) loss for both the training set and validation set across every epoch.
•	Looking at your logs, around Epoch 10, your train MSE was 0.001414 and val MSE was 0.000547. By the time early stopping kicked in at Epoch 68, they had stabilized downward.
Right Plot: Actual vs. Predicted Prices
Python
ax = axes[1]
ax.plot(actual, label="Actual")
ax.plot(predicted, label="Predicted", linestyle="--")
•	This maps the true historical closing prices of Apple (AAPL) in the test set against the LSTM's next-day predictions.
•	The linestyle="--" draws the predictions as a dashed line so you can easily spot if the model's path aligns with or lags behind the actual stock movements.
________________________________________
2. How to Diagnose the Saved Chart (lstm_results.png)
When you open the saved image from your Colab files directory, you want to look for specific visual cues:
Graphic 1: Training Curve (Left)
•	Healthy Training: Both lines should slope steeply downward and flatten out together.
•	Overfitting Warning: If the Train MSE continuous to drop toward zero, but the Val MSE starts curving upward, it means the model is over-optimizing to the training data. Your notebook's Early Stopping safety net successfully cut the execution short at Epoch 68 precisely to prevent this split.
Graphic 2: Test Set: Actual vs. Predicted (Right)
•	The "Lag" Effect: In stock forecasting, LSTMs frequently mimic the shape of the actual price line but shifted slightly to the right by exactly one day. This happens because the easiest mathematical way for the model to minimize MSE loss is to assume "tomorrow's price will be roughly whatever today's price was."
•	Evaluation: Your 67.2% Directional Accuracy indicates that the model is doing significantly better than a simple one-day lag copycat (which usually hovers closer to 50% on direction tracking). The dashed line should track the macro-trends of the solid line cleanly with an average distance error of just $2.32 (MAE).

MAIN
The main() function serves as the master orchestrator of your entire pipeline. It pieces together every single component you have built—from downloading raw financial data to evaluating the optimized LSTM model.
________________________________________
1. Step-by-Step Data Flow in main()
Here is how the data flows sequentially when main() is executed:
 yfinance Data Download ──> Feature Engineering ──> Sequential Window Split (70/10/20)
                                                                  │
   Evaluation Metrics   <──   Test Predictions  <──  Optimized Model Training (fit)
________________________________________
2. Deep Dive Into the Code
Phase A: Setup and Data Preparation
Python
print(f"Device: {DEVICE}")
df = download_data(TICKER, START_DATE, END_DATE)
df = add_features(df)
1.	Hardware Verification: Checks whether a CUDA-capable GPU is available to accelerate the matrix mathematics. Your console output confirmed Device: cuda, which drastically speeds up LSTM training times.
2.	Data Acquisition: Pulls raw daily pricing numbers for Apple (AAPL) from Yahoo Finance between 2018 and 2024.
3.	Indicator Calculation: Passes that dataframe into your feature engine to generate your 13 technical features (like RSI, MACD, and Bollinger Band widths).
________________________________________
Phase B: Processing Data for Deep Learning
Python
(X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = split_and_scale(
    df, SEQ_LEN, TRAIN_RATIO, VAL_RATIO
)
•	Chronological Splitting: Because stock data is inherently dependent on time, main() cannot shuffle the data randomly. It chunks it sequentially into:
o	Training Set (70%): 982 historical rolling windows.
o	Validation Set (10%): 88 historical rolling windows (used for hyperparameter adjustments and early stopping).
o	Test Set (20%): 239 historical rolling windows (held strictly in isolation to evaluate real-world trading performance).
•	Feature Scaling: Uses a MinMaxScaler fitted strictly on the training partition to map all indicators uniformly between 0 and 1.
Python
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
•	PyTorch DataLoaders: Wraps your data arrays into iterable streams. The train_loader uses shuffle=True so that the model doesn't overfit to specific consecutive calendar dates during optimization.
________________________________________
Phase C: Model Architecture Setup

. LSTM (Long Short-Term Memory)
•	Family: RNN (Recurrent Neural Network).
•	Data Type: Sequential data / Time-series (where order matters over time).
•	How it thinks: It processes data step-by-step chronologically and has a "memory cell" to remember past information.
•	Your Notebook: This is why you are using it for Apple stock prices. It reads Day 1, Day 2, up to Day 60 to predict Day 61. It tracks time.

Python
model = LSTMPredictor(
    input_size   = len(FEATURE_COLS),
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT,
).to(DEVICE)
•	Instantiation: Configures your custom LSTMPredictor structure to accept 13 features, map them through a 2-layer deep stacked LSTM architecture with 128 hidden neurons, and drop out 20% of node weights dynamically.
•	.to(DEVICE): Explicitly moves the network's mathematical weights into the GPU VRAM. Your logs indicate this architecture creates exactly 205,441 trainable parameters.
________________________________________
Phase D: Execution, Testing, and Artifact Saving
Python
history = fit(model, train_loader, val_loader, EPOCHS, LR)
•	Training Loop Execution: Fires up the training engine for a ceiling limit of 100 epochs. As seen in your console logs, the validation loss stopped improving over a 15-epoch window, causing the loop to call an early cessation at Epoch 68.
Python
pred_scaled, actual_scaled = predict(model, test_loader)
actual, predicted = report_metrics(actual_scaled, pred_scaled, scaler)
plot_results(history, actual, predicted, TICKER)
•	Testing Phase: Passes the untouched test sequences into the trained model to calculate raw predictions.
•	Denormalization: Takes those fractional outputs and un-scales them back into human-readable Dollar metrics.
•	Reporting & Saving: Prints your final model performance metrics (showing a $2.32 MAE error bound and 67.2% Directional Accuracy) and locks down the training diagnostic graphs into a permanent file named lstm_results.png.



Sumery modal deep learning lstm
It is completely normal to feel confused about this! Time-series deep learning can feel like a "black box." Let’s demystify exactly what the model is doing under the hood with a concrete example.
Think of the LSTM model not as a mathematical equation, but as an experienced Wall Street analyst who has a 60-day short-term memory.
Here is exactly what happens when data goes into the model:
________________________________________
1. What the Model "Sees" (The Input)
The model does not look at the entire history of Apple stock at once. Instead, it looks at a sliding 60-day window.
For every single day inside that window, you feed the model 13 clues (Technical Indicators):
•	Price Clues: Open, High, Low, Close
•	Momentum Clues: RSI (is it overbought?), MACD (is trend shifting?), Returns (how fast did it rise?)
•	Volume Clues: How many shares were traded?
The Input Matrix: If a batch goes in, the model is looking at a grid of data that says: "Here is everything that happened to Apple stock from Day 1 to Day 60."
________________________________________
2. What Happens Inside the LSTM (The Processing)
This is where the math happens. As the data flows through the LSTM layers, two things are happening simultaneously:
•	Step-by-Step Reading: The model reads Day 1, updates its internal "memory notes," then reads Day 2, updates its notes, and repeats this all the way to Day 60.
•	The "Forget" and "Remember" Gates: If Day 5 had a massive volume spike that doesn't matter anymore, the LSTM mathematically "forgets" it. If Day 58 showed a sharp RSI drop, the LSTM "remembers" it as highly relevant context.
By the time it finishes reading Day 60, the LSTM has compressed all 60 days of chaotic stock data into a single vector of 128 numbers (the hidden_size). This vector represents the "vibe" or mathematical summary of the stock's current state.
________________________________________
3. Slicing and the Final Guess (The Output)
In your code, you have a line that says out[:, -1, :]. This is the most crucial part of understanding the "Many-to-One" setup:
Day 1 ──> Day 2 ──> ... ──> Day 59 ──> Day 60 
  │         │                 │          │
 (Ignore)  (Ignore)          (Ignore)   (Keep This Summary Vector)
                                         │
                                   Linear Layer (FC)
                                         │
                                  [ Single Prediction ] ($185.50)
•	The model calculates internal states for Day 1, Day 2, etc., but the code ignores them.
•	It grabs only the very last timestep (-1), which belongs to Day 60. Why? Because Day 60's memory state already contains the accumulated history of the 59 days before it.
•	It passes those 128 numbers into a final Fully Connected (Linear) layer. This layer acts like a final decision-maker, crushing those 128 abstract features down into a single number: The predicted closing price for Day 61.
________________________________________
4. Summary of the Lifecycle
1.	Input: You give it 60 days of 13 stock market features.
2.	LSTM Layer: It scans chronologically, building a 128-number structural context summary of the trend.
3.	Slicing: It throws away the intermediate steps and keeps only the final day's comprehensive summary.
4.	Linear Layer: It translates that final summary into a single Dollar prediction for tomorrow.
5.	Goal: During training, if its guess for Day 61 is $180, but the actual price was $182.32, the optimizer tweaks the internal weights so next time its guess is closer.


Results:
Here is a clear, step-by-step breakdown of how your LSTMPredictor model operates on your stock data under the hood.
________________________________________
1. Structural Setup (__init__)
When you initialize the model, you are configuring three primary mathematical layers:
•	The LSTM Stack (nn.LSTM): Your network is set up with an input_size of 13 (matching your 13 technical features like RSI, MACD, Volume, etc.) and a hidden_size of 128. Because num_layers=2, PyTorch stacks two distinct LSTM networks on top of each other. This allows the deeper layer to extract more abstract, complex long-term historical patterns.
•	The Dropout Layer (nn.Dropout): Configured at 20% (0.2). During training, it randomly switches off 20% of the neurons. This forces the network to find multiple independent paths to solve the prediction problem instead of memorizing a single trend (overfitting).
•	The Linear Layer (nn.FC): A fully connected projection layer. It takes the multi-dimensional internal hidden state of the LSTM and maps it down to a single numeric value—the predicted next-day scaled closing price.
________________________________________
2. Processing Data Flow (forward)
When a mini-batch of 32 sequences passes through the model during training or evaluation, it executes the following logic:
    60-Day Lookback Sequence (32, 60, 13)
                     │
              [ LSTM Layer 1 ]
                     │
              [ LSTM Layer 2 ]
                     │
        Full Sequence Outputs (32, 60, 128)
                     │
      Grab Last Timestep Only: out[:, -1, :] ──> Discards Days 1-59 context
                     │
         Last Day Vector (32, 128)
                     │
              [ Dropout (20%) ]
                     │
            [ Linear Layer (FC) ]
                     │
         Next-Day Prediction (32, 1)
The Slicing Step Explained
The code statement out = self.dropout(out[:, -1, :]) represents a Many-to-One architecture setup.
The LSTM layer initially outputs a 128-dimensional context vector for every single day in your 60-day window. However, because you are only interested in predicting the single price for the upcoming day (Day 61), the intermediate states from Days 1 to 59 are sliced away.
The model selects only the very last timestep (-1). Because an LSTM processes sequences chronologically, the memory state at Day 60 already inherently carries the accumulated context of the preceding 59 days.
________________________________________
3. Your Live Training Summary
According to the console logs from your execution:
•	The model processed 205,441 trainable parameters.
•	Early Stopping triggered automatically at Epoch 50 because your validation loss ceased to improve, ensuring the network didn't overfit to your training subset.
•	On the test set, the model achieved a Mean Absolute Error (MAE) of $2.01, meaning its predictions deviated from the actual historical Apple price by an average of just over two dollars.
•	It achieved a Directional Accuracy of 69.3%, correctly predicting whether the price would go up or down tomorrow roughly 7 out of 10 times.






