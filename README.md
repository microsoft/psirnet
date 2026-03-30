# Step-by-Step Guide to Training a PSIRNet
## Tyger Setup
- Follow tyger instructions and `tyger-config.yml` under the `configs` directory. Common pitfalls include using the default `apiAppUri` and `cliAppUri`, as well as forgetting `serviceManagementReference: 7933f882-e109-49d5-a771-e1be70099806`. The choice between east vs. west is important. Check it before proceeding.

## Reconstruct NIH Data Archive
- To reconstruct NIH Cardiac MRI Raw Data Archive, use the `nihrawdata` repository. Ideally have a csv file for which files to reconstruct. You need a ManagedIdentity which has read access to the storage account. 
```
loader create-buffers -f /workspaces/nihrawdata/retrocine_LGE.csv --identity=lge-reader
loader recon -c /workspaces/nihrawdata/overrides.yml -d -f /workspaces/nihrawdata/retrocine_LGE.csv
loader recon -c /workspaces/nihrawdata/overrides.yml -f /workspaces/nihrawdata/retrocine_LGE.csv --rps 10
```

## Trainining, Validation, and Test
- To train models using `amlt`, follow the `resys` repository (`resy-sing-setup`) and the CPU devcontainer with the following addition to the `devcontainer.json` to ensure mount to the project directory:
```
	"mounts": [
		"source=/home/t-aatalik/psirnet,target=/workspaces/psirnet,type=bind,consistency=cached"
	]
```
- Once a project is successfully set up, go to Azure Portal and check the ServicePrincipal. In case there are multiple, look for the Azure Resource ID and ensure the provider is `Microsoft.ManagedIdentity`. For example, the project `psirnet` has the following Azure Resource ID:
```
/subscriptions/87d8acb3-5176-4651-b457-6ab9cefd8e3d/resourcegroups/psirnet/providers/Microsoft.ManagedIdentity/userAssignedIdentities/psirnet
```
  <img width="428" height="310" alt="Screenshot 2025-08-08 162943" src="https://github.com/user-attachments/assets/42e3952d-7ebc-45a5-9b76-a1dd4408600a" />

  In the image above:
  - ApplicationID is used in `tyger login` from a job (e.g., `tyger login https://lge-tyger.eastus.cloudapp.azure.com --identity --identity-client-id 1e77c3f8-f251-4a24-be33-0792b45d1551`)
  - ObjectID is used in `tyger` configuration file. After adding these IDs, run `tyger access-control apply`, `tyger cloud install` and `tyger api install`.
  ```
  - kind: ServicePrincipal
    objectId: 74ff1b4a-4278-4e03-88a4-fa4b5c388bc7
    displayName: psirnet
  ```
  Now singularity jobs should have access to tyger buffers, so that training / validation can use a streaming approach as in the codebase. GCR images might require access for directly calling them in a job. One simple solution is `docker pull` from GCR registry and `docker push` to the project's.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
