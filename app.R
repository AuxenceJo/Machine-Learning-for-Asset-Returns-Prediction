# Shiny Application for Financial ML Analysis
# ==============================================

# Load required packages
library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(DT)
library(plotly)
library(tidyverse)
library(ranger)
library(xgboost)
library(Metrics)
library(caret)
library(glmnet)
library(keras)
library(corrplot)
library(lubridate)
library(shinycssloaders)
library(dplyr)

# Increase max file upload size since default Shiny limit is 5MB
options(shiny.maxRequestSize = 30*1024^2)  # 30 MB

# ============================================
# UTILITY FUNCTIONS
# ============================================

# Data cleaning function
# Removes columns with excessive missing values, converts dates, filters out invalid rows, and imputes missing numeric values with medians
clean_data <- function(data, na_threshold = 0.1) {
  missing_data <- colSums(is.na(data))
  valid_cols <- names(missing_data[missing_data/nrow(data) < na_threshold])
  
  df_clean <- data %>%
    select(stock_id, date, all_of(valid_cols)) %>%
    mutate(date = as.Date(date)) %>%
    filter(!is.na(R1M_Usd)) %>%
    mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .))) %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
  
  return(df_clean)
}

# Feature engineering
# Creates new features like z-scores, ratios, and time-based variables to enhance model performance
create_features <- function(data) {
  df_features <- data %>%
    group_by(stock_id) %>%
    arrange(date) %>%
    mutate(
      Mom_5M_Z = (Mom_5M_Usd - mean(Mom_5M_Usd, na.rm = TRUE)) / sd(Mom_5M_Usd, na.rm = TRUE),
      Vol_Z = (Vol1Y_Usd - mean(Vol1Y_Usd, na.rm = TRUE)) / sd(Vol1Y_Usd, na.rm = TRUE),
      Sharp_Vol_Ratio = Mom_Sharp_11M_Usd / (Vol1Y_Usd + 1e-8),
      Mom_Delta = Mom_5M_Usd - lag(Mom_5M_Usd, n = 1)
    ) %>%
    ungroup() %>%
    mutate(
      month = month(date),
      quarter = quarter(date),
      year = year(date)
    ) %>%
    filter(!is.na(Mom_Delta))
  
  return(df_features)
}

# Strategy evaluation function
# Evaluates long-short strategy performance based on model predictions, calculating returns and Sharpe ratio
evaluate_strategy <- function(predictions, actual_returns, top_pct = 0.2, bottom_pct = 0.2) {
  df_ranks <- tibble(
    prediction = predictions,
    actual = actual_returns
  ) %>%
    mutate(rank_pct = percent_rank(prediction))
  
  top <- df_ranks %>% filter(rank_pct >= (1 - top_pct))
  bottom <- df_ranks %>% filter(rank_pct <= bottom_pct)
  
  if(nrow(top) == 0 || nrow(bottom) == 0) {
    return(list(ls_return = 0, ls_sharpe = 0, top_return = 0, bottom_return = 0))
  }
  
  perf <- list(
    top_return = mean(top$actual, na.rm = TRUE),
    bottom_return = mean(bottom$actual, na.rm = TRUE),
    ls_return = mean(top$actual, na.rm = TRUE) - mean(bottom$actual, na.rm = TRUE),
    top_vol = sd(top$actual, na.rm = TRUE),
    bottom_vol = sd(bottom$actual, na.rm = TRUE)
  )
  
  perf$ls_vol <- sqrt((perf$top_vol^2 + perf$bottom_vol^2) / 2)
  perf$ls_sharpe <- ifelse(perf$ls_vol > 0, perf$ls_return / perf$ls_vol, 0)
  
  return(perf)
}

# ============================================
# USER INTERFACE
# ============================================

ui <- dashboardPage(
  dashboardHeader(title = "Financial ML Analysis"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Exploration", tabName = "exploration", icon = icon("chart-line")),
      menuItem("Modeling", tabName = "modeling", icon = icon("robot")),
      menuItem("Strategies", tabName = "strategies", icon = icon("coins")),
      menuItem("Results", tabName = "results", icon = icon("trophy"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        .box {
          border-radius: 10px;
        }
        .info-box-text {
          font-size: 14px;
        }
      "))
    ),
    
    tabItems(
      # ===== DATA TAB =====
      # Technical Details: Allows users to upload RData files, set cleaning parameters, and preview cleaned data
      tabItem(tabName = "data",
              fluidRow(
                box(
                  title = "Section Overview", status = "info", solidHeader = TRUE, width = 12,
                  HTML("
    <h4> Data Upload & Cleaning</h4>
    <p>This section allows you to upload your financial dataset (<code>.RData</code>), clean it, and preview the result.</p>
    <ul>
      <li>Remove columns with excessive missing values.</li>
      <li>Impute <code>NA</code> values with column medians.</li>
      <li>Remove <code>Inf</code> values and invalid rows.</li>
      <li>Filter out observations where <code>R1M_Usd</code> is missing.</li>
      <li>Standardize date format to <code>Date</code>.</li>
    </ul>
    <p><em>This ensures clean, consistent inputs for downstream modeling.</em></p>
  ")
                )
                              ),
              fluidRow(
                box(
                  title = "Data Upload", status = "primary", solidHeader = TRUE, width = 12,
                  fileInput("file", "Choose RData file",
                            accept = c(".RData", ".rdata")),
                  verbatimTextOutput("data_info")
                )
              ),
              
              fluidRow(
                box(
                  title = "Data Cleaning Parameters", status = "info", solidHeader = TRUE, width = 6,
                  sliderInput("na_threshold", "Missing Values Threshold (%)",
                              min = 5, max = 50, value = 10, step = 5),
                  actionButton("clean_data", "Clean Data", class = "btn-warning")
                ),
                
                box(
                  title = "Data Preview", status = "success", solidHeader = TRUE, width = 6,
                  withSpinner(DT::dataTableOutput("data_preview"))
                )
              )
      ),
      
      # ===== EXPLORATION TAB =====
      # Technical Details: Provides visualizations including target distribution, correlation matrix, and time series analysis
      tabItem(tabName = "exploration",
              fluidRow(
                box(
                  title = "Section Overview", status = "info", solidHeader = TRUE, width = 12,
                  HTML("
    <h4> Exploratory Analysis</h4>
    <p>This section provides tools to explore your dataset:</p>
    <ul>
      <li>Histogram of <code>R1M_Usd</code> to check distribution & outliers.</li>
      <li>Correlation matrix to detect multicollinearity.</li>
      <li>Time series plots to observe temporal patterns.</li>
    </ul>
    <p><em>These tools help uncover structure in the data and guide feature selection.</em></p>
  ")
                )
                              ),
              fluidRow(
                box(
                  title = "Target Variable Distribution", status = "primary", solidHeader = TRUE, width = 6,
                  withSpinner(plotlyOutput("target_dist"))
                ),
                
                box(
                  title = "Correlation Matrix", status = "primary", solidHeader = TRUE, width = 6,
                  withSpinner(plotOutput("correlation_matrix"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Time Series Evolution", status = "info", solidHeader = TRUE, width = 12,
                  selectInput("time_var", "Variable to Visualize:",
                              choices = NULL),
                  withSpinner(plotlyOutput("time_series"))
                )
              )
      ),
      
      # ===== MODELING TAB =====
      # Technical Details: Allows training of multiple ML models (Lasso, Random Forest, XGBoost) with configurable train/test split and correlation threshold
      tabItem(tabName = "modeling",
              fluidRow(
                box(
                  title = "Section Overview", status = "info", solidHeader = TRUE, width = 12,
                  HTML("
    <h4> ML Training</h4>
    <p>Train models to predict future returns:</p>
    <ul>
      <li>Select models: Lasso, Random Forest, XGBoost, Neural Network.</li>
      <li>Define train/test split and correlation threshold for feature reduction.</li>
      <li>Evaluate using:</li>
        <ul>
          <li><strong>R²</strong>: proportion of variance explained</li>
          <li><strong>RMSE</strong>: root mean squared error</li>
        </ul>
    </ul>
    <p><em>Models are trained on historical features to learn return dynamics.</em></p>
    
    <p><em>⏳ Please note: Model training may take between 5 to 7 minutes depending on data size and selected models.</em></p>
  ")
                )
                
                
              ),
              fluidRow(
                box(
                  title = "Modeling Parameters", status = "primary", solidHeader = TRUE, width = 4,
                  sliderInput("train_ratio", "Train/Test Ratio (%)",
                              min = 60, max = 90, value = 80, step = 5),
                  sliderInput("cor_threshold", "Correlation Threshold",
                              min = 0.5, max = 0.95, value = 0.8, step = 0.05),
                  checkboxGroupInput("models", "Models to Train:",
                                     choices = list(
                                       "Lasso" = "lasso",
                                       "Random Forest" = "rf",
                                       "XGBoost" = "xgb",
                                       "Neural Network" = "nn"
                                     ),
                                     selected = c("lasso", "rf", "xgb")),
                  actionButton("train_models", "Train Models", 
                               class = "btn-success", style = "width: 100%;")
                ),
                
                box(
                  title = "Training Progress", status = "info", solidHeader = TRUE, width = 8,
                  withSpinner(verbatimTextOutput("training_progress")),
                  br(),
                  withSpinner(DT::dataTableOutput("model_performance"))
                )
              )
      ),
      
      # ===== STRATEGIES TAB =====
      # Technical Details: Evaluates long-short strategies based on model predictions, with configurable top/bottom percentiles
      tabItem(tabName = "strategies",
              fluidRow(
                box(
                  title = "Section Overview", status = "info", solidHeader = TRUE, width = 12,
                  HTML("
    <h4> Strategy Evaluation</h4>
    <p>Backtest long-short strategies from model predictions:</p>
    <ul>
      <li>Rank predictions.</li>
      <li>Long top <code>x%</code>, short bottom <code>y%</code>.</li>
      <li>Evaluate strategy via average returns and Sharpe ratio</li>
    </ul>
    
    <p><em>Cumulative returns show robustness over time.</em></p>
  ")
                )
                              ),
              fluidRow(
                box(
                  title = "Strategy Parameters", status = "primary", solidHeader = TRUE, width = 4,
                  sliderInput("top_pct", "% Top Positions", 
                              min = 5, max = 30, value = 20, step = 5),
                  sliderInput("bottom_pct", "% Bottom Positions",
                              min = 5, max = 30, value = 20, step = 5),
                  selectInput("strategy_model", "Model for Strategy:",
                              choices = NULL),
                  actionButton("evaluate_strategy", "Evaluate Strategy", class = "btn-info")
                ),
                
                box(
                  title = "Strategy Performance", status = "success", solidHeader = TRUE, width = 8,
                  withSpinner(DT::dataTableOutput("strategy_performance"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Temporal Performance", status = "info", solidHeader = TRUE, width = 12,
                  withSpinner(plotlyOutput("cumulative_returns"))
                )
              )
      ),
      
      # ===== RESULTS TAB =====
      # Technical Details: Summarizes model and strategy performance with R² comparisons, quintile analysis, and a downloadable report
      tabItem(tabName = "results",
              fluidRow(
                box(
                  title = "Section Overview", status = "info", solidHeader = TRUE, width = 12,
                  HTML("
    <h4> Model & Strategy Results</h4>
    <p>This section summarizes model and strategy performance:</p>
    <ul>
      <li>Display the best model (based on Sharpe).</li>
      <li>Compare models by R², RMSE, Sharpe, and return.</li>
      <li>Visualize return by quintile.</li>
      <li>Download a full report.</li>
    </ul>
    <p><em>Use these insights to refine models and select robust strategies.</em></p>
  ")
                )
              ),
              fluidRow(
                valueBoxOutput("best_model"),
                valueBoxOutput("best_sharpe"),
                valueBoxOutput("total_return")
              ),
              
              fluidRow(
                box(
                  title = "Performance Comparison (R²)", status = "primary", solidHeader = TRUE, width = 6,
                  plotlyOutput("performance_barplot")
                ),
                
                box(
                  title = "Quintile Analysis", status = "success", solidHeader = TRUE, width = 6,
                  withSpinner(plotlyOutput("quintile_analysis"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Final Report", status = "info", solidHeader = TRUE, width = 12,
                  downloadButton("download_report", "Download Report", class = "btn-primary"),
                  br(), br(),
                  withSpinner(verbatimTextOutput("final_report"))
                )
              )
      )
    )
  )
)

# ============================================
# SERVER
# ============================================

server <- function(input, output, session) {
  
  # Reactive variable for training progress logs
  training_log <- reactiveVal("") # initial value
  
  # Utility function to add a log message
  log_message <- function(msg) {
    old_log <- training_log()
    timestamp <- format(Sys.time(), "%H:%M:%S")
    new_log <- paste(old_log, paste0("[", timestamp, "] ", msg), sep = "\n")
    training_log(new_log)
  }
  
  # Display logs in the "Training Progress" box
  output$training_progress <- renderText({
    training_log()
  })
  
  # Reactive values
  values <- reactiveValues(
    raw_data = NULL,
    clean_data = NULL,
    features_data = NULL,
    train_data = NULL,
    test_data = NULL,
    models = list(),
    predictions = NULL,
    strategies = NULL
  )
  
  # Utility function for progress logging (duplicate removed)
  
  output$performance_barplot <- renderPlotly({
    req(values$models)
    
    df <- map_dfr(names(values$models), function(name) {
      tibble(Model = name, R2 = values$models[[name]]$r2)
    })
    
    p <- ggplot(df, aes(x = Model, y = R2, fill = Model)) +
      geom_col() +
      labs(title = "R² by Model", y = "R²") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # ===== DATA LOADING =====
  observeEvent(input$file, {
    req(input$file)
    
    tryCatch({
      load(input$file$datapath)
      values$raw_data <- data_ml  # Assume the variable is named data_ml
      
      output$data_info <- renderText({
        paste("Data loaded:",
              "\n- Dimensions:", nrow(values$raw_data), "x", ncol(values$raw_data),
              "\n- Period:", min(values$raw_data$date, na.rm = TRUE), "to", max(values$raw_data$date, na.rm = TRUE),
              "\n- Unique stocks:", length(unique(values$raw_data$stock_id)))
      })
      
    }, error = function(e) {
      showNotification("Error loading file", type = "error")
    })
  })
  
  # ===== DATA CLEANING =====
  observeEvent(input$clean_data, {
    req(values$raw_data)
    
    withProgress(message = "Cleaning data...", {
      tryCatch({
        values$clean_data <- clean_data(values$raw_data, input$na_threshold / 100)
        values$features_data <- create_features(values$clean_data)
        
        # Update visualization choices
        numeric_vars <- names(values$features_data)[sapply(values$features_data, is.numeric)]
        updateSelectInput(session, "time_var", choices = numeric_vars, selected = "R1M_Usd")
        
        # Notification with shinyWidgets
        shinyWidgets::sendSweetAlert(
          session = session,
          title = "Cleaning Complete",
          text = "Data cleaned successfully!",
          type = "success"
        )
      }, error = function(e) {
        showNotification(paste("Error during cleaning:", e$message), type = "error")
      })
    })
  })
  
  # ===== DATA PREVIEW =====
  output$data_preview <- DT::renderDataTable({
    req(values$clean_data)
    DT::datatable(head(values$clean_data, 1000), options = list(scrollX = TRUE))
  })
  
  # ===== EXPLORATORY VISUALIZATIONS =====
  output$target_dist <- renderPlotly({
    req(values$features_data)
    
    p <- ggplot(values$features_data, aes(x = R1M_Usd)) +
      geom_histogram(bins = 50, fill = "steelblue", alpha = 0.8) +
      geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
      labs(title = "Distribution of 1-Month Returns",
           x = "Return (%)", y = "Frequency") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  output$correlation_matrix <- renderPlot({
    req(values$features_data)
    
    main_vars <- c("R1M_Usd", "Mom_5M_Usd", "Mom_11M_Usd", 
                   "Vol1Y_Usd", "Mkt_Cap_12M_Usd", "Mom_Sharp_5M_Usd")
    
    available_vars <- intersect(main_vars, names(values$features_data))
    
    if(length(available_vars) >= 3) {
      corr_matrix <- cor(values$features_data[, available_vars], use = "complete.obs")
      corrplot(corr_matrix, method = "color", type = "upper", 
               order = "hclust", tl.cex = 0.8, tl.col = "black")
    }
  })
  
  output$time_series <- renderPlotly({
    req(values$features_data, input$time_var)
    
    time_data <- values$features_data %>%
      group_by(date) %>%
      summarise(value = mean(.data[[input$time_var]], na.rm = TRUE)) %>%
      arrange(date)
    
    p <- ggplot(time_data, aes(x = date, y = value)) +
      geom_line(color = "steelblue") +
      labs(title = paste("Evolution of", input$time_var),
           x = "Date", y = input$time_var) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  observeEvent(input$train_models, {
    req(values$features_data, input$models)
    
    withProgress(message = "Training models...", {
      
      tryCatch({
        incProgress(0.1, detail = "Preparing data...")
        log_message("📊 Preparing data...")
        
        df <- values$features_data %>% arrange(date)
        cutoff_index <- floor(nrow(df) * input$train_ratio / 100)
        cutoff_date <- df$date[cutoff_index]
        
        values$train_data <- df %>% filter(date < cutoff_date)
        values$test_data <- df %>% filter(date >= cutoff_date)
        
        all_numeric <- names(df)[sapply(df, is.numeric)]
        feature_names <- setdiff(all_numeric, c("stock_id", "R1M_Usd", "R3M_Usd", "R6M_Usd", "R12M_Usd"))
        
        if (length(feature_names) > 1) {
          feature_cor <- cor(df[, feature_names], use = "complete.obs")
          high_cor <- findCorrelation(feature_cor, cutoff = input$cor_threshold, verbose = FALSE)
          reduced_features <- feature_names[-high_cor]
        } else {
          reduced_features <- feature_names
        }
        
        X_train <- values$train_data %>% select(all_of(reduced_features))
        y_train <- values$train_data$R1M_Usd
        X_test <- values$test_data %>% select(all_of(reduced_features))
        y_test <- values$test_data$R1M_Usd
        
        results <- list()
        model_list <- input$models
        total_models <- length(model_list)
        start_global <- Sys.time()
        
        for (i in seq_along(model_list)) {
          model <- model_list[i]
          start_time <- Sys.time()
          progress_pct <- round((i - 1) / total_models * 100)
          
          log_message(paste0("⚙️ Starting training ", model, " (", progress_pct, "%)..."))
          
          if (model == "lasso") {
            incProgress(1 / total_models, detail = "Lasso")
            tryCatch({
              cv_lasso <- cv.glmnet(as.matrix(X_train), y_train, alpha = 1, nfolds = 3)
              model_lasso <- glmnet(as.matrix(X_train), y_train, alpha = 1, lambda = cv_lasso$lambda.min)
              pred_lasso <- predict(model_lasso, as.matrix(X_test), s = cv_lasso$lambda.min)
              
              results$lasso <- list(
                model = model_lasso,
                predictions = as.vector(pred_lasso),
                r2 = 1 - sum((y_test - pred_lasso)^2) / sum((y_test - mean(y_test))^2),
                rmse = sqrt(mean((y_test - pred_lasso)^2))
              )
            }, error = function(e) {
              showNotification("Error with Lasso", type = "warning")
              log_message("❌ Lasso Error")
            })
          }
          
          if (model == "rf") {
            incProgress(1 / total_models, detail = "Random Forest")
            tryCatch({
              rf_model <- ranger(
                formula = R1M_Usd ~ .,
                data = cbind(R1M_Usd = y_train, X_train),
                num.trees = 100,
                importance = "permutation"
              )
              pred_rf <- predict(rf_model, data = X_test)$predictions
              
              results$rf <- list(
                model = rf_model,
                predictions = pred_rf,
                r2 = 1 - sum((y_test - pred_rf)^2) / sum((y_test - mean(y_test))^2),
                rmse = sqrt(mean((y_test - pred_rf)^2))
              )
            }, error = function(e) {
              showNotification("Error with Random Forest", type = "warning")
              log_message("❌ Random Forest Error")
            })
          }
          
          if (model == "xgb") {
            incProgress(1 / total_models, detail = "XGBoost")
            tryCatch({
              dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
              dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
              xgb_model <- xgb.train(
                params = list(objective = "reg:squarederror", eta = 0.2, max_depth = 4),
                data = dtrain,
                nrounds = 50,
                verbose = 0
              )
              pred_xgb <- predict(xgb_model, newdata = dtest)
              
              results$xgb <- list(
                model = xgb_model,
                predictions = pred_xgb,
                r2 = 1 - sum((y_test - pred_xgb)^2) / sum((y_test - mean(y_test))^2),
                rmse = sqrt(mean((y_test - pred_xgb)^2))
              )
            }, error = function(e) {
              showNotification("Error with XGBoost", type = "warning")
              log_message("❌ XGBoost Error")
            })
          }
          
          end_time <- Sys.time()
          duration <- round(difftime(end_time, start_time, units = "secs"), 1)
          elapsed_global <- difftime(end_time, start_global, units = "secs")
          est_remaining <- (as.numeric(elapsed_global) / i) * (total_models - i)
          est_min <- floor(est_remaining / 60)
          est_sec <- round(est_remaining %% 60)
          
          log_message(paste0("✅ ", model, " completed in ", duration, " sec. Progress: ",
                             round(i / total_models * 100), "%. Estimated time remaining: ",
                             est_min, " min ", est_sec, " sec."))
        }
        
        values$models <- results
        updateSelectInput(session, "strategy_model", choices = names(results), selected = names(results)[1])
        showNotification("Models trained successfully!", type = "message")
        
      }, error = function(e) {
        showNotification(paste("Global error:", e$message), type = "error")
      })
    })
  })
  
  # ===== MODEL PERFORMANCE =====
  output$model_performance <- DT::renderDataTable({
    req(values$models)
    
    perf_df <- map_dfr(names(values$models), function(name) {
      tibble(
        Model = name,
        R2 = round(values$models[[name]]$r2, 4),
        RMSE = round(values$models[[name]]$rmse, 4)
      )
    })
    
    DT::datatable(perf_df, options = list(dom = 't'))
  })
  
  # ===== STRATEGY EVALUATION =====
  observeEvent(input$evaluate_strategy, {
    req(values$models, input$strategy_model)
    
    withProgress(message = "Evaluating strategies...", {
      
      tryCatch({
        strategies <- map_dfr(names(values$models), function(name) {
          predictions <- values$models[[name]]$predictions
          actual <- values$test_data$R1M_Usd
          
          strat <- evaluate_strategy(predictions, actual, 
                                     input$top_pct/100, input$bottom_pct/100)
          
          tibble(
            Model = name,
            `L-S Return` = round(strat$ls_return * 100, 2),
            `Sharpe Ratio` = round(strat$ls_sharpe, 3),
            `Top Return` = round(strat$top_return * 100, 2),
            `Bottom Return` = round(strat$bottom_return * 100, 2)
          )
        })
        
        values$strategies <- strategies
        showNotification("Strategies evaluated!", type = "message")
        
      }, error = function(e) {
        showNotification(paste("Error during evaluation:", e$message), type = "error")
      })
    })
  })
  
  output$strategy_performance <- DT::renderDataTable({
    req(values$strategies)
    DT::datatable(values$strategies, options = list(dom = 't'))
  })
  
  output$cumulative_returns <- renderPlotly({
    req(values$models, input$strategy_model)
    
    model_name <- input$strategy_model
    req(model_name %in% names(values$models))
    
    predictions <- values$models[[model_name]]$predictions
    actual_returns <- values$test_data$R1M_Usd
    dates <- values$test_data$date
    
    # Calculate cumulative long/short return
    df <- tibble(date = dates,
                 prediction = predictions,
                 actual = actual_returns) %>%
      group_by(date) %>%
      mutate(rank = percent_rank(prediction)) %>%
      summarise(
        top = mean(actual[rank >= 0.8], na.rm = TRUE),
        bottom = mean(actual[rank <= 0.2], na.rm = TRUE),
        ls = top - bottom
      ) %>%
      mutate(cum_return = cumsum(ls))
    
    p <- ggplot(df, aes(x = date, y = cum_return)) +
      geom_line(color = "darkblue") +
      labs(title = paste("Cumulative L/S Return -", model_name),
           x = "Date", y = "Cumulative Return") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  output$feature_importance <- renderPlotly({
    req(values$models)
    
    if ("rf" %in% names(values$models)) {
      imp <- values$models$rf$model$variable.importance
      imp_df <- tibble(
        Feature = names(imp),
        Importance = as.numeric(imp)
      ) %>%
        arrange(desc(Importance)) %>%
        slice(1:20)  # Top 20
      
      p <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_col(fill = "steelblue") +
        coord_flip() +
        labs(title = "Feature Importance (Random Forest)",
             x = "Feature", y = "Importance") +
        theme_minimal()
      
      ggplotly(p)
    }
  })
  
  output$quintile_analysis <- renderPlotly({
    req(values$models, input$strategy_model)
    
    model_name <- input$strategy_model
    preds <- values$models[[model_name]]$predictions
    actuals <- values$test_data$R1M_Usd
    
    df <- tibble(pred = preds, actual = actuals) %>%
      mutate(quintile = ntile(pred, 5)) %>%
      group_by(quintile) %>%
      summarise(mean_return = mean(actual, na.rm = TRUE))
    
    p <- ggplot(df, aes(x = factor(quintile), y = mean_return)) +
      geom_col(fill = "darkgreen") +
      labs(title = paste("Quintile Analysis -", model_name),
           x = "Prediction Quintile", y = "Average Return") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # ===== VALUE BOXES =====
  output$best_model <- renderValueBox({
    if(is.null(values$strategies)) {
      valueBox("N/A", "Best Model", icon = icon("trophy"), color = "blue")
    } else {
      best <- values$strategies[which.max(values$strategies$`Sharpe Ratio`), ]
      valueBox(best$Model, "Best Model", icon = icon("trophy"), color = "blue")
    }
  })
  
  output$best_sharpe <- renderValueBox({
    if(is.null(values$strategies)) {
      valueBox("N/A", "Best Sharpe", icon = icon("chart-line"), color = "green")
    } else {
      best_sharpe <- max(values$strategies$`Sharpe Ratio`, na.rm = TRUE)
      valueBox(round(best_sharpe, 3), "Best Sharpe", icon = icon("chart-line"), color = "green")
    }
  })
  
  output$total_return <- renderValueBox({
    if(is.null(values$strategies)) {
      valueBox("N/A", "Best Return", icon = icon("coins"), color = "yellow")
    } else {
      best_return <- max(values$strategies$`L-S Return`, na.rm = TRUE)
      valueBox(paste0(best_return, "%"), "Best Return", icon = icon("coins"), color = "yellow")
    }
  })
  
  # ===== FINAL REPORT =====
  output$final_report <- renderText({
    if(is.null(values$strategies) || is.null(values$models)) {
      return("Train models and evaluate strategies first.")
    }
    
    paste(
      "===== SUMMARY REPORT =====",
      paste("Analysis period:", min(values$test_data$date), "to", max(values$test_data$date)),
      paste("Number of test observations:", nrow(values$test_data)),
      "",
      "MODEL PERFORMANCE:",
      paste(capture.output(print(values$strategies)), collapse = "\n"),
      "",
      paste("Best model:", values$strategies$Model[which.max(values$strategies$`Sharpe Ratio`)]),
      paste("Optimal Sharpe ratio:", round(max(values$strategies$`Sharpe Ratio`, na.rm = TRUE), 3)),
      sep = "\n"
    )
  })
}

# ============================================
# LAUNCH THE APPLICATION
# ============================================

shinyApp(ui = ui, server = server)