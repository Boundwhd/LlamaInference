#ifndef _OP_ADD_H_
#define _OP_ADD_H_
#include "layer.h"
namespace op {
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);
  
    base::Status check() const override;
  
    base::Status forward() override;
};
}
#endif