Full Stack Web Engineer Exercise
================================

Overview
--------

This exercise requires the engineer to perform a range of tasks that are typical in developing, deploying, and operating a live site. 

The project is a blank slate, the only artifact being the products.json file, which will form the basis for the application's data model.

The exit criteria of this exercise is to observe:

* What method and technology was used to deploy core infrastructure like databases, web servers, etc.
* What technologies were selected and why?
* What architecture patterns were selected and why?

The over arching goal of this exercise is to gain insight into how the engineer approaches building and deploying a full stack project. Given the limited time box of most of our explorations, a strong grasp of current tools and frameworks crucial to rapidly building prototypes capable of semi-production deployment is key.

Scenario
--------

This exercise is based on a simple end to end scenario; building a basic product viewer web application.

We have a raw product file, products.json, and the following requirements:

1. That a user can view the product images and titles through a web application
2. That an application developer can access the data through an API
3. That a user can search for a product by entering a keyword on the web application
4. That an operations person can view basic graphs on the load/use of the application

Requirements
------------

Your code should meet the following requirements:

* Be [PEP8] compliant
* Implement a unit test for each material function, capable of being tested using [pytest] or [nosetest]
* Contain sufficient docstrings and comments to make it easy to reason about


Things to keep in mind
----------------------
* Full stack encompasses everything from design, development, deployment and operations
* This is a mile wide, inch deep exercise, focus on clearing all tasks at the same level, and avoid going deeper than need be into any one task

Submission Process
------------------

To avoid dependencies on services like AWS, we suggest you complete your work inside a [Docker] container, using Ubuntu 13.04 (64 bit) or 13.10 (64 bit) as the base OS, and simply check-in your application's docker file as part of your overall Github fork.

Disclaimer
---------

Merchant product listings in data/products.json are the property of the respective merchant.

[PEP8]:http://legacy.python.org/dev/peps/pep-0008/
[pytest]:http://pytest.org/latest/
[nosetest]:https://nose.readthedocs.org/en/latest/
[Docker]:https://www.docker.io/

