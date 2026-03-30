export $(cat .env | xargs)
EXPERIMENT_NAME="PSIRNet_$(date +%Y%m%d_%H%M%S)"
amlt run -f -y ${AMLT_CONFIG} ${EXPERIMENT_NAME} -t ${TARGET}