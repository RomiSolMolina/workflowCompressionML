/*
 * globals.h
 *
 *  Created on: Jun 16, 2023
 *      Author: ivan
 */

#ifndef SRC_GLOBALS_H_
#define SRC_GLOBALS_H_


#include "comblock.h"
#include "xmyproject.h"
#include "pulses.h"

/*
 * ComBlock-related defines
 */

#define COMBLOCK_0	XPAR_COMBLOCK_0_AXIL_BASEADDR
#define CB_REG_SAMPLE	CB_OREG0




/*
 * ML HLS Inference block defines
 */

XMyproject mlInferenceInstancePtr;
XMyproject_Config *mlInferenceCfgPtr;





/*
 * Prototypes
 */
int initPeripherals(void);
int feedPulse(unsigned int *pulseType, unsigned int pulseIndex);




#endif /* SRC_GLOBALS_H_ */
