# Intel® Distribution of OpenVINO™ HDDL-R Setup Guide*

This guide includes instructions for:
1.	Installing the required IEI* BSL reset driver.
2.	Configuration settings for the hddldaemon service.

**NOTE**: This guide does not apply to Uzel* cards.

## IEI HDDL-R Reset Driver Installation

Using an IEI HDDL-R board like the Mustang V100-MX8* requires downloading and installing the most current driver for your system.

Visit the [IEI Download Center](https://download.ieiworld.com/) for the latest drivers and to search for software and documentation.
Search for:     **Mustang-V100-MX8**

As of 8/13/2019, the following links were for the latest drivers:

[IEI Reset Driver Download, Linux](https://dls.ieiworld.com/IEIWeb/PDC_APP/PLM/OWFP000225/Mustang-V100_Linux_Plugin_1.0.1.20190409.tar.gz)

[IEI Reset Driver Download, Windows](https://dls.ieiworld.com/IEIWeb/PDC_APP/PLM/OWFP000225/Mustang-V100_win64_Plugin_1.0.1.20190409.7z)

Download the appropriate driver for your system, decompress the downloaded archive, enter the newly created directory, and run the install script:

On **Linux***:
-  Run the `install.sh script` with `sudo`, or as `root`.

On **Windows***, do one of the following:<br>
-  **GUI**: Double-click `install.bat`
-  **CLI**: Open a console with administrator privileges, cd into the directory, and run `install.bat`.

## HDDL-R Service Configuration

The `hddldaemon` is a system service, a binary executable that is run to manage the computational workload on the HDDL-R board.  It’s a required abstraction layer that handles inference, graphics processing, and any type of computation that should be run on the VPUs (video processing units).  Depending on the board configuration, there will be 8 or 16 VPUs on the board.

**NOTE**: Graphics and other specialized computation may require some custom development.

The following lists a few of the possible configuration options.  
DAVID: See this non-existent reference document for complete details. 

### Conventions Used in this Document
`<IE>` refers to the following default OpenVINO Inference Engine Directories:
-  **Linux:**	    `/opt/intel/openvino/inference_engine`
-  **Windows:**	    `C:\Program Files (x86)\IntelSWTools\openvino\inference_engine`

If you have installed OpenVINO in a different directory on your system, you will need to enter your unique directory path.

### Configuration File Location

`<IE>\external\hddl\config\hddl_service.config`

### Service Configuration File Settings

**NOTE:**  After changing a configuration file, the HDDL service must be restarted. 

### Settings of Interest:

`device_snapshot_mode`<br>
Changes the output of the HDDL service to display a table with individual VPU statistics.

**Default Setting:**<br>	
  `"device_snapshot_mode":    "none"`

**Suggested Setting:**<br>	
  `"device_snapshot_mode":    "full"`

**Supported Settings:**
-  `none` (Default)
-  `base`
-  `full`

`device_snapshot_style`

**Default Setting:**<br>	
  `"device_snapshot_style":    "table"`

**Recommended Setting:**<br>	
  `"device_snapshot_style":    "table"`<br>
The `table` setting is cleaner and presents labels on the left for each column.  <br>The `tape` setting prints the labels in each column.

**Supported Settings:**
-  `tape`
-  `table` (default)


`user_group	`<br>	
Restricts the service to group members. 

**Recommended	    TBD for individual use cases (user or service)**

**Default Setting**	,<br>
  `“user_group”:    "users"`

The HDDL-R service may be restricted to a privileged group of users.  The appropriate group will vary according to the local system configuration.

**Supported Settings:**<br>
Valid groups on the current system.  The `“users”` group is a default group that exists on Windows and most Linux distributions.


**Optional Recommended Settings:**

`"device_utilization" : “off”`<br>
This setting displays the percent of time each VPU is in use.  It appears in the `table` column when turned on, or if `“device_fps”` is turned on.

`“memory_usage” : “off”`<br>
This setting reports the amount of memory being used by each VPU.

`“max_cycle_switchout”: 3`<br>
Requires the squeeze scheduler.  This setting might speed up performance significantly, depending on the app.  

**NOTE:** This setting works in conjunction with: `max_task_number_switch_out`.

`"client_fps" : “off”`<br>
This setting reports the total FPS for the dispatching hddl_service (which will have one or more clients per app).

`debug_service`<br>
`“debug_service”: “false”`<br>	(Default: `“true”`)

## HDDLR Programming Reference

The following section provides information on how to distribute a model across all 8 VPUs to maximize performance.

## Programming a C++ Application for the HDDL-R

**Declare a Structure to Track Requests**

The structure should hold:
1.	A pointer to an inference request.
2.	An ID to keep track of the request.<br>

    `struct Request {`<br>
        `InferenceEngine::InferRequest::Ptr inferRequest;`<br>
        `int frameidx;`<br>
    `};`

**Declare a Vector of Requests:**

    `// numRequests` is the number of frames (max size, equal to the number of VPUs in use).
    `vector<Request> request(numRequests);` 

Declare and initialize 2 mutex variables:
1.	For each request.
2.	For when all 8 requests are done.

**Declare a Conditional Variable:** (indicates when at most 8 requests are done at a time)

For inference requests, use the asynchronous IE API calls:<br>

    `// initialize infer request pointer` – Consult IE API for more detail.
    `request[i].inferRequest = executable_network.CreateInferRequestPtr();`

    `// Run inference`<br>
    `request[i].inferRequest->StartAsync();`


**Create a Lambda Function:** (for the parsing and display of results)<br>
Inside the Lambda body use the completion callback function:<br>

    `request[i].inferRequest->SetCompletionCallback`<br>
    `(nferenceEngine::IInferRequest::Ptr context)`


## Programming a Python Application for the HDDL-R

[DAVID] TBD will add after receiving source from Aleks Sutic.


## Additional Resources

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)

- [Intel Distribution of OpenVINO Toolkit documentation](https://docs.openvinotoolkit.org)

