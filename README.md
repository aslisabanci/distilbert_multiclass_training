# Preventing Model Drift with Continuous Monitoring and Deployment using Github Actions and Algorithmia Insights

[Model governance](https://algorithmia.com/blog/model-governance) is an essential component of an organization’s overall ML framework, especially when machine learning models get to production. Rules and processes around model management, immutability, and observability gain more importance when companies are at this stage and maturity level. Besides deploying models to production through automated and traceable processes, keeping an eye and acting on model drift in production and making sure that models perform accurately over time becomes crucial. 

Advocating for these best practices, we talked about how you can continuously train and deploy your machine learning models from your Jupyter notebook to Algorithmia, with our automated workflows on Github Actions [in our previous blog post](https://algorithmia.com/blog/how-to-continuously-deploy-models-from-an-ml-repository-to-algorithmia-with-github-actions). 

With the recent release of our mighty [Algorithmia Insights](https://algorithmia.com/blog/algorithmia-insights-press-release), we can now funnel algorithm metrics to a Kafka topic and monitor that insights stream using a preferred monitoring application. 

In this blog post, we will take things one step further and combine how you can turn these insights into actionable items. We will talk about how you can trigger your machine learning model’s re-training and re-deployment using Github Actions when your model’s monitored metrics deteriorate. With this integration in place, you will see how you can build a new version of your algorithm at Algorithmia throughout a rule based automation process; and fix your model metrics in production by re-deploying your improved model. 

As usual, let’s walk over an example use-case to show each part of this integration.

## Overview

We build this example around a multiclass text classifier model, which is a fine-tuned DistilBERT model using `transformers` and `PyTorch`.  

Our NLP model’s training repo at Github uses a [CI/CD](https://github.com/marketplace/actions/deploy-to-algorithmia) automation to run its training notebook and deploy the created model file and the serving layer scripts to Algorithmia by [our Continuous Deployment Github Action](https://github.com/marketplace/actions/deploy-to-algorithmia). 

To serve this model on Algorithmia’s GPU workers, we have an “algorithm” created on a base environment pre-configured with CUDA drivers and a PyTorch installation. 

Using Algorithmia Insights, we collect our algorithm’s metrics and pass them to a Kafka topic. Our Algorithmia-Datadog connector forwards the collected metrics to Datadog over Kafka and we setup monitors on Datadog to observe our model’s metrics.

. ![](images/usecase.png)

Piece by piece, we will show you how we take automated action upon a simulated model drift, trigger model re-training through our Github Actions and have a newer version of the model deployed to Algorithmia, throughout a rule-based chain of automations. 



## Training data on S3

Our model’s training data set sits under an AWS S3 bucket, as a CSV file. This is a combination of user reviews for 3 topics: Food, clothing and movies; and it contains both positive and negative reviews. So our model is fine-tuned to classify text inputs into 6 categories: Food - Positive, Food - Negative, Clothing - Positive, Clothing - Negative, Movies - Positive, Movies - Negative.



![](images/training_data_s3.png)



For productionized ML applications, it’s a usual practice to continuously collect and label new training data and store these on cloud buckets. ML practitioners use these new ground truth data for their future training runs or to work on developing different variations of their models. This practice keeps the ML models “fresh” and robust with more and up-to-date data.

So when our automated workflow is triggered, our model training notebook accesses this S3 bucket to fetch the most recent training data, downloads it on the runner machine, and trains the model with it.



## Model training repo at Github

We create our [The model training repository at Github](https://github.com/aslisabanci/distilbert_multiclass_training/) based on our [“model deployment template” repository](https://github.com/algorithmiaio/githubactions-modeldeployment-template), as mentioned on our [previous blog post](https://algorithmia.com/blog/how-to-continuously-deploy-models-from-an-ml-repository-to-algorithmia-with-github-actions). This template comes with stub files, directories and an automated workflow file, pre-configured to use Algorithmia’s continuous model deployment Github Action. 

Following this practice, our example training repo contains:

- Github Actions workflow file, to continuously train and deploy models to Algorithmia

- Jupyter notebook to train the fine-tuned DistilBERT model by fetching the data from the S3 bucket, training the model and exporting the checkpoints

- Algorithm files (inference script and its requirements) to be deployed to Algorithmia

- Requirements file for the training notebook


  ![](images/github_repo.png)

  

  ## Configuring Github Actions workflow

  In case this is the first time you’re reading about Github Actions, you can check out our [previous blog post]((https://algorithmia.com/blog/how-to-continuously-deploy-models-from-an-ml-repository-to-algorithmia-with-github-actions)) for an introduction to our [Continuous Deployment to Algorithmia workflow](https://github.com/algorithmiaio/algorithmia-modeldeployment-action) and learn more about them. You can follow that hands-on tutorial and see how easy and fun it is to use it for your own training repositories and models at Algorithmia!

  Otherwise you should already be familiar with the configuration below. One thing that stands out from our previous example is the `git_host` input, pointing to a private cluster. This is the case for our enterprise customers, but of course if you’re using Algorithmia on our public marketplace, you can leave off that parameter for its default value.

  ```yaml
  name: Deploy to Algorithmia
  
  on:
    workflow_dispatch:
    push:
      branches:
        - master
  
  jobs:
    algorithmia-deploy:
      runs-on: self-hosted
      steps:
        - name: Checkout Main Repo
          uses: actions/checkout@v2.0.0
          with:
            ref: ${{github.sha}}
            path: main
        - name: Deploy Model to Algorithmia
          uses: algorithmiaio/algorithmia-modeldeployment-action@master
          env:
            HEAD_COMMIT_MSG: ${{ github.event.head_commit.message }}
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
            AWS_DEFAULT_REGION: ${{ secrets.AWS_DEFAULT_REGION }}
          with:
            algorithmia_api_key: ${{ secrets.ALGORITHMIA_API_KEY }}
            algorithmia_username: ${{ secrets.ALGORITHMIA_USERNAME }}
            algorithmia_email: ${{ secrets.ALGORITHMIA_EMAIL }}
            algorithmia_password: ${{ secrets.ALGORITHMIA_PASSWORD }}
            algorithmia_algoname: distilbert_multiclass_classifier
            git_host: git.devopsbay1.enthalpy.click
            model_path: distilbert-checkpoints.pth
  ```

  

  ## ## Running workflows on a self-hosted GPU runner

  Since we want to train our model on a GPU, we’re using a self-hosted GPU machine as our [CI/CD]((https://github.com/marketplace/actions/deploy-to-algorithmia)) runner. To prepare our GPU machine to run the assigned workflows, we go to our repository’s **Settings** -> **Actions**, click on the **Add runner** button and follow Github’s instructions while we’re connected to it through `ssh` .

  Once we finish the setup, we see our GPU machine named `deeppurple` listed under the Runners section, tagged with some labels. Lastly, you must have noticed that in the workflow file above, the `runs-on` key’s value is `self-hosted`. So you now know the reason!

  

  ![](images/githubactions_selfhosted_configured.png)

  

  ## Testing Github Actions deployment for initial verification

  Looking at the top of the workflow definition, you can see that we configure the workflow to get triggered either with a `workflow_dispatch` event or a `push` on the `master` branch. The `workflow_dispatch` is for us to trigger this via Github API, when we’re configuring our model confidence alert.

  After starting the action job listener on our self-hosted GPU runner, we can test that the training and the deployment works as expected, by just pushing to our master branch. In a couple of seconds, we see that the GPU runner gets the job, starts executing it and when the job succeeds, the trained and exported model is uploaded to a data source at Algorithmia along with the algorithm scripts and we have a new build of the algorithm with a new hash.

  This is exactly what will happen when our model drifts on production, creating an alert, and we trigger model re-training. So we’re slowly getting there, keep reading!![](images/github_to_algorithmia.png)

  ## DistilBERT algorithm at Algorithmia

  To serve our text classification model behind a scalable REST API, we have an algorithm at Algorithmia. 

  The “algorithm” is an encapsulation of the machine learning model and its serving layer duo. It accepts authenticated HTTP requests with the input data as a JSON blob and returns the predictions of the model in the HTTP response.

  This algorithm loads our DistilBERT checkpoints from an Algorithmia data source, parses and processes the input data in the incoming requests, feeds the preprocessed input to the machine learning model and returns the predictions of the model in the response body.

  

  ![](images/algorithm.png)

  ![](images/algorithm_input_output.png)

  

  ## Enabling Algorithmia Insights

  Here comes the gist of this use-case, the [algorithm insights](https://algorithmia.com/blog/algorithmia-insights-press-release)! Insights is a new feature of Algorithmia Enterprise, for instrumenting metrics from the machine learning models and algorithms hosted at Algorithmia. So besides the basic serving functionality, our algorithm collects and transmits insights about our model’s behavior.To learn more and configure Algorithmia Insights, you can check out [our step-by-step walkthrough](https://algorithmia.com/blog/how-to-use-algorithmia-insights-for-machine-learning-model-performance-monitoring) and get it up and running in just a few minutes. ## Sending insights for model’s confidence scoreIn this use case, we want to observe and monitor our model’s prediction confidence. This metric tells us how confident our model is, when it says the given text belongs to a certain class. To represent this confidence level, we will use a score between 0 and 1. Let’s take a quick look at how we send this confidence score to this Kafka broker over Algorithmia Insights, with the `report_insights` method of our Algorithmia client library.

  ```python
  def predict(input_texts):
    	#Preparing the input texts for the model
      encoded_inputs = tokenizer.batch_encode_plus(
          input_texts,
          max_length=250,
          padding=True,
          truncation=True,
          return_token_type_ids=False,
      )
      input_ids = torch.tensor(encoded_inputs["input_ids"])
      attention_masks = torch.tensor(encoded_inputs["attention_mask"])
  		
      # Turning them into digestible tensors
      tensor_ds = TensorDataset(input_ids, attention_masks)
      data_loader = DataLoader(
          tensor_ds, sampler=SequentialSampler(tensor_ds), batch_size=32
      )
      
      #Getting the predictions from the model
      classes, probs = evaluate(data_loader)
  
      results = []
      for predicted_class, prob in zip(classes, probs):
          #Mapping the predicted numbers to our 6 classes' user-friendly names
          predicted_class = class_mapping[int(np.argmax(predicted_class))]
          #Getting the highest predicted probability to use for confidence
          confidence = float(np.max(prob))
          results.append(
              {"class": predicted_class, "confidence": confidence,}
          )
          #Sending model metrics over Algorithmia Insights
          client.report_insights({"model_confidence": confidence})
      return results
  ```

  

  As you see, it’s just a one-liner and we can add more to these pipelined metrics depending on what we want to measure and monitor.

  

  ## Forwarding model insights to Datadog

  For this example, we will use Datadog as our monitoring application to observe our model metrics. 

  To make this connection, we have a detailed walk-through on [how to monitor model performance metrics with Datadog](https://algorithmia.com/blog/algorithmia-integration-how-to-monitor-model-performance-metrics-with-datadog) and you can also refer to the [integration page on Datadog](https://docs.datadoghq.com/integrations/algorithmia/) and [documentation in the Algorithmia integrations repository](https://github.com/algorithmiaio/integrations/tree/master/Datadog) for more detailed information on installing and configuring this integration. 

  

  ## Observing model confidence on Datadog

  After getting setup with the integration, we can customize our default dashboards or add new ones. So we have the following dashboard widgets on Datadog to observe our model’s confidence score over time, monitor the minimum and the average values within a time window. 

  

  ![](images/datadog_okay_dashboard.png)

  

  ## Monitoring for model confidence warning and alerts

  In order to get notified for warnings and alerts, we also have monitors in place, for `algorithmia.model_confidence` metric, averaged by `algorithm_name`.For this example, we configured Datadog to warn us when the `model_confidence` crosses the value warning threshold, set as 0.8; and alert us when it crosses the alert threshold set as 0.7. 

  ![](images/datadog_monitor_config.png)

  

  ## Integrating Datadog with Slack

  We connect our Slack account with Datadog, to receive messages on our `classification-confidence` channel, when the model confidence crosses the warning threshold. ![](images/datadog_slack_integration.png)

  ## Integrating Datadog with Github Actions

  In order to trigger our Github Actions workflow for re-training and re-deploying our model, we set up a webhook integration. Here we’re using the Github API to send a POST request to our repository’s workflow endpoint, with the shown payload and the headers. 

  

  ![](images/datadog_webhooks_integration.png)

  

  Let’s go over each section here, so that you can also set up this integration for your own repository and workflow files. 

  **URL** field points to https://api.github.com/repos/aslisabanci/distilbert_multiclass_training/actions/workflows/algorithmia_deploy.yml/dispatches for our example. So this would be http://api.github.com/repos/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/YOUR_WORKFLOW_FILENAME.yml/dispatches for your own setup.

  The **PAYLOAD** section contains a JSON body and it tells Github which branch to run this workflow on. 

  ```json
  {
    "ref": "master"
  }
  ```

  

  **CUSTOM HEADERS** section also contains our headers in JSON format. For your setup, you should use your own Github Access Token in the Authorization header:

  ```json
  {
      "Accept": "application/vnd.github.v3+json", 
      "Authorization": "token YOUR_GITHUB_ACCESS_TOKEN",
      "Content-Type": "application/x-www-form-urlencoded"
  }
  ```

  And finally, do not check the **ENCODE AS FORM** option.  

  

  ## Setting up notifications for warnings and alerts

  After integrating Datadog with our Slack channel and the automated workflow endpoint, now it’s time to configure how we want to trigger these integrations. This last piece of the configuration may be the easiest of all, because we already have every piece ready and it’s just a matter of chaining those now. So for this last piece, we just embody the handles of the integrations with `#is_alert` and `#is_warning` tags, decorate it with an informative caption and that’s it!

  

  ![](images/datadog_monitor_notifications.png)

  

  

  ## Time to test!

  Here comes the most enjoyable part: Seeing everything work end-to-end! ![](images/integration_components.png)

  ### Testing warnings

  Let’s test our algorithm by sending it some text inputs that we know our model isn’t able to classify with a high confidence. When this is the case, our algorithm output looks like below, with confidence above 0.7 but not around 0.9s or higher.

  Since we configured our monitor to warn us when model confidence drops below 0.8 threshold, we see this warning on Datadog:

  ![](images/datadog_warning_triggered.png)and we get a Slack notification on the previously configured **#classification-confidence** channel right after!

  

  Looking at the notification message body, we can see which algorithm is causing this alert and a graph of the most recent values for this observed metric. 

  ![](images/slack_warning_triggered.png)

  

  If you’ve added Datadog app to your Slack account, you can even select this message on Slack, click on **More Actions -> More Message Shortcuts...** and create a ticket on Jira with the enclosed information. This way you can turn this warning into an actionable item and acknowledge, prioritize, and track it by putting it on the record. 

  ![](images/slack_jira.png)

  

  ### Testing Alerts

  When model confidence drops even further, the alert monitor gets triggered and we see this Alert on Datadog:

  ![](images/datadog_alert_triggered.png)

  

  This alert triggers the Github Actions webhook for our automated workflow and voila, our self-hosted GPU runner that was listening for incoming jobs gets a job assigned by Github, starts running our automated workflow configuration, named `algorithmia-deploy`

  ![](images/selfhosted_running_ci.png)

  So we kick off the re-training of our model with our most recent training data on the S3 bucket. Once the training is finished and the model is exported from the Jupyter notebook, the new model will be deployed to Algorithmia and handle the incoming requests. 

  

  ### Observing drift recovery after model re-training and deployment

  Assuming that the newly deployed model is an improved version of its predecessor as it’s trained with the most recently collected training data, we should see our model’s confidence rising again to the expected levels and our monitors returning to the “OK” status on our dashboard. 

  

  ![](images/datadog_recover.png)

  

  ## Get started with Algorithmia Insights and continuous deployment

  We hope you’ve enjoyed following along with this use case and these example integrations help you with your model governance processes. We’d love to know if you have any questions, comments, or feedback to improve these model deployment integrations even further

  

  ![](images/PreventingDriftWithAlgorithmia.m4v)