# AKLR

AKLR serves GPU kernels to inference libraries by taking over some of their functions. This glossary fixes the words used for that integration.

## Language

### Plugins

**Host** (宿主):
One of the libraries AKLR serves kernels to: sglang, vllm or aiter.
_Avoid_: target library, framework

**Target** (目标):
A host function AKLR may serve, named by its dotted full name; in aiter, by the name of the op aiter routes.
_Avoid_: api, plugined api, route

**Declaration** (声明):
The statement that a target is described by a contract on given host versions.

**Family** (族):
The contracts serving one target, or a few related targets, kept together with their kernels.

**Register function** (注册函数):
The function a host's plugin mechanism calls at start-up, through which AKLR takes over the declared targets.
_Avoid_: hook function

**Interceptor** (截获函数):
The function standing in for a target once AKLR has taken it over.
_Avoid_: hook, route function, selector

**Host version** (宿主版本):
The version a declaration is validated against: a release version for sglang and vllm, an op version for aiter.
_Avoid_: version (unqualified)

**Fallback** (回退):
Running the host's own implementation because no AKLR kernel serves the call.

### Kernels

**Device target** (设备目标):
The GPU architecture and hardware a kernel is built for. Always said with "device": a bare "target" is a host function.
_Avoid_: target (unqualified)
