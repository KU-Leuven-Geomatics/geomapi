.. Geomapi documentation master file, created by
   sphinx-quickstart on Thu May  5 11:46:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GEOMAPI's documentation!
===================================

.. image:: ../pics/week23-uav4.PNG
   :alt: Overview data
   :width: 800px
   :height: 300px
   :align: center

Welcome to GEOMAPI, a Python toolbox designed to enhance the processing of close-range sensing observations such as images and point clouds of the built environment. It specializes in jointly utilizing `Building Information Modeling (BIM) <https://en.wikipedia.org/wiki/Building_information_modeling>`_ and sensory data. To achieve this, GEOMAPI employs `Linked Data (LD) <https://en.wikipedia.org/wiki/Linked_data>`_ to create resources of various popular sensory modalities.

If you want to learn more about GEOMAPI's structure, head over to the  `Getting started <testcases/Getting started>`_ section. Similarly, you can check out the `Ontology <testcases/Ontology>`_ to better understand the data management. If you're interested in viewing real test cases solved with GEOMAPI code, visit the `TEST CASES <testcases/alignmenttools>`_ section.

This `Open-Source <https://github.com/KU-Leuven-Geomatics/geomapi>`_ API is the work of the `GEOMATICS <https://iiw.kuleuven.be/onderzoek/geomatics>`_ research group at KU Leuven, Belgium. If you want to collaborate, visit the team section and let us know.

.. toctree::
   :maxdepth: 1
   :caption: Information:

   information/getting started

   information/nodes

   information/functionality

   information/ontology

.. toctree::
   :maxdepth: 1
   :caption: API Reference   

   geomapi/geomapi.nodes

   geomapi/geomapi.utils

   geomapi/geomapi.tools

.. toctree::
   :maxdepth: 1
   :caption: Tutorial:

   tutorial/tutorial_nodes

   tutorial/tutorial_geometrynodes

   .. tutorial/tutorial_meshnodes

   .. tutorial/tutorial_pointcloudnodes

   .. tutorial/tutorial_bimnodes

   tutorial/tutorial_imagenodes

   tutorial/tutorial_sessionnodes

   tutorial/tutorial_node_selection

.. toctree::
   :maxdepth: 1
   :caption: Test Cases:

   testcases/alignmenttools
   
   testcases/combinationtools

   testcases/site_progress

   testcases/validationtools

   testcases/volume_calculation

.. toctree::
   :maxdepth: 1
   :caption: Development:

   development/environment creation

   development/packaging

   development/ontology creation

   development/testcase creation

.. toctree::
   :maxdepth: 1
   :caption: Contribution

   team/team
   `GitHub <https://github.com/KU-Leuven-Geomatics/geomapi>`_

Please do refer to the following publication when using GEOMAPI in your projects. 

.. image:: ../pics/paper.PNG
   :target: https://www.sciencedirect.com/science/article/pii/S0926580524001900
   :width: 200px
   :height: 300px
   :alt: GEOMAPI Paper

.. code-block:: none
@article{GEOMAPI,
            title = {GEOMAPI: Processing close-range sensing data of construction scenes with semantic web technologies},
            journal = {Automation in Construction},
            volume = {164},
            pages = {105454},
            year = {2024},
            issn = {0926-5805},
            doi = {https://doi.org/10.1016/j.autcon.2024.105454},
            url = {https://www.sciencedirect.com/science/article/pii/S0926580524001900},
            author = {Maarten Bassier and Jelle Vermandere and Sam De Geyter and Heinder De Winter},
            keywords = {Geomatics, Semantic Web Technologies, Construction, Close-range sensing, BIM, Point clouds, Photogrammetry}
            }



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
