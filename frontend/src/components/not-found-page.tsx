import { Link } from "react-router-dom";
import { Result, Button } from "antd";

export default function NotFoundPage() {
  return (
    <Result
      status="404"
      title="Page not found"
      subTitle="We couldn't find the page you requested."
      extra={
        <Link to="/">
          <Button type="primary">Go home</Button>
        </Link>
      }
    />
  );
}
