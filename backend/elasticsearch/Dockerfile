# Elasticsearch with Nori (Korean morphological analyzer) plugin
FROM docker.elastic.co/elasticsearch/elasticsearch:8.14.0

# Install Nori plugin for Korean text analysis
RUN elasticsearch-plugin install --batch analysis-nori

# Verify installation
RUN elasticsearch-plugin list | grep analysis-nori
